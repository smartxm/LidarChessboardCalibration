import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np


class LidarGuiCalibrator:
    def __init__(self, sub_pcd, plane, cam_parmams):
        self.original_pcd = sub_pcd
        self.pcd = None

        self.grid_points = None
        self.plane = plane
        self.cam_parmams = cam_parmams

        self.offset = 0.005  # 调试红线向摄像头方向偏移距离（避免被点云遮挡）可以调

        self.corner_points = []   # 存4个角点（世界坐标）
        self.pick_mode = True     # 是否在选点模式
        """
        plane.center   # 平面中心
        plane.R        # 3x3 旋转矩阵
        R[:, 0]  # 第1列 → 平面局部 x 轴
        R[:, 1]  # 第2列 → 平面局部 y 轴
        R[:, 2]  # 第3列 → 法向量（normal）
        """

    # ================= 平面拟合 + 投影 =================
    def ProjectToPlane(self):
        """把点云投影到平面上"""

        points = np.asarray(self.original_pcd.points)

        # ===== 平面信息 =====
        center = self.plane.center
        R = self.plane.R
        normal = R[:, 2]   # 法向量

        # ===== 投影 =====
        projected = []

        for p in points:
            v = p - center
            dist = np.dot(v, normal)     # 点到平面的距离（带符号）
            p_proj = p - dist * normal   # 投影
            projected.append(p_proj)

        projected = np.array(projected)

        # ===== 构造新的点云 =====
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(projected)

        # 保留颜色（如果有）
        if self.original_pcd.has_colors():
            self.pcd.colors = self.original_pcd.colors

    def init_scene(self):
        """创建窗口、场景，并初始化相机"""

        app = gui.Application.instance

        # ===== 创建窗口 =====
        window = app.create_window("Lidar Calibrator", 1024, 768)

        # ===== 创建 SceneWidget =====
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)

        window.add_child(scene)

        # ===== 相机参数 =====
        center, eye, up = self.cam_parmams

        # 创建并初始化一个 OrbitCamera（轨道相机）
        scene.setup_camera(
            60.0,
            self.pcd.get_axis_aligned_bounding_box(),
            center
        )

        # 必须先初始化轨道相机再设置相机参数才能生效！
        scene.scene.camera.look_at(center, eye, up)

        # ===== 背景 & 光照 =====
        scene.scene.set_background([1, 1, 1, 1])
        scene.scene.scene.set_sun_light([1, 1, 1], [1, 1, 1], 75000)
        scene.scene.scene.enable_sun_light(True)

        # 添加点云
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # 设置点的大小
        mat.point_size = 5.0
        scene.scene.add_geometry("pcd", self.pcd, mat)
        # ===== 添加网格 =====
        grid, grid_points = self.create_grid_lines()
        self.grid_points = grid_points

        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 1.0

        scene.scene.add_geometry("grid", grid, mat_line)

        # scene.set_on_mouse(self.on_mouse)
        # self.scene = scene

        return window, scene



    def create_grid_lines(self):
        """在平面上生成10x7网格线"""

        R = self.plane.R
        center = self.plane.center

        x_axis = R[:, 0]
        y_axis = R[:, 1]

        # 转换到平面局部坐标
        points = np.asarray(self.pcd.points)
        local_pts = []

        for p in points:
            v = p - center
            x = np.dot(v, x_axis)
            y = np.dot(v, y_axis)
            local_pts.append([x, y])

        local_pts = np.array(local_pts)

        xmin, ymin = local_pts.min(axis=0)
        xmax, ymax = local_pts.max(axis=0)
        # xmin = xmin*0.9
        # ymin = ymin*0.9
        # xmax = xmax*0.9
        # ymax = ymax*0.9
        # 生成分割线
        x_lines = np.linspace(xmin, xmax, 11)[1:-1]  # 9条竖线
        y_lines = np.linspace(ymin, ymax, 8)[1:-1]   # 6条横线

        # 构建线
        points_3d = []
        lines = []
        colors = []

        # 法向量修正
        normal = self.plane.R[:, 2]
        # 计算从平面指向相机的方向
        plane_center = self.plane.center
        eye = np.asarray(self.cam_parmams[1])
        view_dir = eye - plane_center
        view_dir = view_dir / np.linalg.norm(view_dir)
        # 点积判断方向
        if np.dot(normal, view_dir) < 0:
            normal = -normal

        def to_world(x, y):
            base = center + x * x_axis + y * y_axis
            return base + self.offset * normal

        # 竖线
        for x in x_lines:
            p1 = to_world(x, ymin)
            p2 = to_world(x, ymax)

            idx = len(points_3d)
            points_3d.append(p1)
            points_3d.append(p2)

            lines.append([idx, idx + 1])
            colors.append([1, 0, 0])

        # 横线
        for y in y_lines:
            p1 = to_world(xmin, y)
            p2 = to_world(xmax, y)

            idx = len(points_3d)
            points_3d.append(p1)
            points_3d.append(p2)

            lines.append([idx, idx + 1])
            colors.append([1, 0, 0])

        # 创建 LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set, points_3d

    # ================= 主入口 =================
    def run(self):
        app = gui.Application.instance
        app.initialize()

        self.ProjectToPlane()
        # 初始化场景
        window, scene = self.init_scene()



        # ===== 4️⃣ 运行 =====
        app.run()