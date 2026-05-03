import open3d as o3d
import numpy as np
from lidar_gui_calibrator import LidarGuiCalibrator

def load_csv_to_pointcloud(csv_path):
    """
    从包含 X,Y,Z 的 CSV 文件读取点云并转换为 Open3D PointCloud
    """

    # 读取表头
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')

    x_idx = header.index("X")
    y_idx = header.index("Y")
    z_idx = header.index("Z")
    r_idx = header.index("Reflectivity")  # ⭐新增

    # 读取 XYZ + Reflectivity
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1,
        usecols=(x_idx, y_idx, z_idx, r_idx)  # ⭐注意这里
    )

    points = data[:, :3]
    reflect = data[:, 3]

    # 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 归一化反射率并映射到颜色
    reflect = (reflect - reflect.min()) / (reflect.max() - reflect.min())
    colors = np.stack([reflect, reflect, reflect], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def planar_patch_detection(pcd):
    """检测点云中的平面"""
    # 如果没有法向量 → 计算
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )

    # 检测平面
    planes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=30,   # 更严格
        coplanarity_deg=60,
        outlier_ratio=0.3,
        min_plane_edge_length=0.5,
        min_num_points=500,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50)
    )

    print("检测到平面数:", len(planes))

    return planes

def visualize_planes(pcd,planes):
    """可视化提取到的彩色平面，输出新的渲染队列"""
    geometries = [pcd]
    for obox in planes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
            obox, scale=[1, 1, 0.001]
        )
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)

    return geometries

def adjust_perspective(geometries):
    """第一步：在可以看到已有平面的情况下调整视角"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for g in geometries:
        vis.add_geometry(g)

    vis.get_render_option().point_size = 2.0

    ctr = vis.get_view_control()

    # 相机初始视角
    ctr.set_zoom(0.05)
    ctr.set_front([-2, 0, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0.5, -0.3, 5.2])

    print("调整好视角后按 Q")

    vis.run()

    # ⭐ 保存相机参数
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    return cam_params

def pick_points(geometries,cam_params):
    """第二布：保留上一补的视角，选点
    返回选择的子点云"""
    vis_pick = o3d.visualization.VisualizerWithEditing()
    vis_pick.create_window(window_name="Shift+左键选点")

    for g in geometries:
        vis_pick.add_geometry(g)

    vis_pick.get_render_option().point_size = 2.0

    # ⭐ 恢复相机
    ctr_pick = vis_pick.get_view_control()
    ctr_pick.convert_from_pinhole_camera_parameters(cam_params)

    vis_pick.run()
    vis_pick.destroy_window()

    picked = vis_pick.get_picked_points()

    if len(picked) == 0:
        print("没有选点")
        exit()

    idx = picked[0]
    clicked_point = np.asarray(pcd.points)[idx]

    print("选中的点:", clicked_point)


    # ===== 找对应平面 =====
    def point_to_plane_distance(point, obox):
        normal = obox.R[:, 2]
        center = obox.center
        return abs(np.dot(point - center, normal))


    min_dist = 1e9
    selected_obox = None

    for i, obox in enumerate(planes):
        dist = point_to_plane_distance(clicked_point, obox)
        if dist < min_dist:
            min_dist = dist
            selected_obox = obox
            plane_id = i

    print("选中平面:", plane_id)

    sub_pcd = pcd.crop(selected_obox)

    return sub_pcd, selected_obox

def get_plane_view(sub_pcd):
    """展平子点云，使其平行于电脑屏幕展示，获取需要的初始相机位置参数"""
    # 用 PCA 拟合平面法向量
    # 把 Open3D.PointCloud 转成 N×3 的矩阵
    points = np.asarray(sub_pcd.points)

    # 计算质心
    # numpy.mean()函数用于计算数组元素的算术平均值，axis=0:对每一列求平均（即对points(是array)中所有点的x,y,z分别求平均）
    center = points.mean(axis=0)

    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    # 最小特征值对应的特征向量 = 法向量具体看不太懂怎么算出来的，总之一顿对协方差矩阵做特征值分解，然后是这一行的提取法向量
    normal = eigvecs[:, np.argmin(eigvals)]

    # 统一方向（让它朝向相机）
    # 让法向量朝向原点（或你想看的方向）
    view_point = np.array([0, 0, 0])  # 你希望“正面”朝向的点

    # 用 点积（dot product） 判断方向，该值小于0时法向量背对相机
    if np.dot(normal, view_point - center) < 0:
        normal = -normal

    # 计算一个合理的相机位置（eye）
    bbox = sub_pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())

    distance = extent * 2.0   #  可以调：1.5 ~ 3.0

    eye = center + normal * distance

    # up 向量 
    up = np.array([0, 0, 1])

    if abs(np.dot(up, normal)) > 0.9:
        up = np.array([0, 1, 0])

    return center.tolist(), eye.tolist(), up.tolist()

print("Load a csv point cloud, print it, and render it")
print("Testing IO for point cloud ...")
pcd = load_csv_to_pointcloud("../input/3.csv")
print(pcd)
print(np.asarray(pcd.points))

planes = planar_patch_detection(pcd)

geometries = visualize_planes(pcd,planes)

# ================= 第一阶段：调整视角 =================
cam_params = adjust_perspective(geometries)

# ================= 第二阶段：选点 =================
sub_pcd,plane = pick_points(geometries,cam_params)

# ===== 渲染 =====
center, eye, up = get_plane_view(sub_pcd)

print(center,eye,up)
calib = LidarGuiCalibrator(sub_pcd,plane,(center,eye,up))

# ⭐ 进入GUI（内部不会再重复投影）
calib.run()
