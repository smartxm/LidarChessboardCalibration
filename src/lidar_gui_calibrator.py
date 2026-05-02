import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np


class LidarGuiCalibrator:
    def __init__(self, sub_pcd, plane, cam_parmams):
        self.original_pcd = sub_pcd
        self.pcd = None

        self.plane = plane
        self.cam_parmams = cam_parmams
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

        return window, scene

    # ================= 主入口 =================
    def run(self):
        app = gui.Application.instance
        app.initialize()

        self.ProjectToPlane()
        # 初始化场景
        window, scene = self.init_scene()



        # ===== 4️⃣ 运行 =====
        app.run()