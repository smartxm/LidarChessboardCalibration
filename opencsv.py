import open3d as o3d
import numpy as np

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


print("Load a csv point cloud, print it, and render it")
print("Testing IO for point cloud ...")
pcd = load_csv_to_pointcloud("input/1.csv")

print(pcd)
print(np.asarray(pcd.points))

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.05,
                                  front=[-2, 0, 0],
                                  lookat=[0,0,0],
                                  up=[0.5, 0, 5.2])