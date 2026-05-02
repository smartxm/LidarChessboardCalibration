import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np


class LidarGuiCalibrator:
    def __init__(self, sub_pcd, cam_parmams):
        self.original_pcd = sub_pcd
        self.pcd = None

        self.cam_parmams = cam_parmams

        self.plane_center = None
        self.plane_normal = None

        self.lines = []
        self.points_clicked = []

    # ================= 平面拟合 + 投影 =================
    def prepare_data(self):
        pts = np.asarray(self.original_pcd.points)

        center = pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        normal = eigvecs[:, np.argmin(eigvals)]

        # ⭐ 统一方向（朝向相机）
        front = np.array(self.cam_parmams[1])
        if np.dot(normal, front) < 0:
            normal = -normal

        projected = pts - np.dot((pts - center), normal)[:, None] * normal

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(projected)

        if self.original_pcd.has_colors():
            self.pcd.colors = self.original_pcd.colors

        self.plane_center = center
        self.plane_normal = normal

    # ================= 初始化GUI =================
    def init_scene(self):
        self.window = gui.Application.instance.create_window(
            "Lidar Calibrator", 1024, 768
        )

        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)

        self.window.add_child(self.widget)

        # ===== 添加点云 =====
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0

        self.widget.scene.add_geometry("pcd", self.pcd, mat)

        # ===== 相机初始化（关键替换你原来的 get_plane_view）=====
        self.init_camera()

        # ===== 鼠标 =====
        self.widget.set_on_mouse(self.on_mouse)

    # ================= 相机初始化 =================
    def init_camera(self):
        center = np.array(self.plane_center)
        front  = np.array(self.cam_parmams[1])
        up     = np.array(self.cam_parmams[2])

        # 自动计算距离
        bbox = self.pcd.get_axis_aligned_bounding_box()
        scale = np.linalg.norm(bbox.get_extent())

        eye = center - front * scale * 1.5

        self.widget.scene.camera.look_at(center, eye, up)

    # ================= 鼠标点击 =================
    def on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            if event.is_button_down(gui.MouseButton.LEFT):

                world = self.pick_point(event.x, event.y)
                if world is None:
                    return gui.Widget.EventCallbackResult.IGNORED

                self.points_clicked.append(world)

                if len(self.points_clicked) == 2:
                    self.add_line(
                        self.points_clicked[0],
                        self.points_clicked[1]
                    )
                    self.points_clicked.clear()

                return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    # ================= 屏幕 → 3D =================
    def pick_point(self, x, y):
        def depth_callback(depth_image):
            depth = np.asarray(depth_image)[y, x]

            if depth == 1.0:
                self.last_point = None
                return

            world = self.widget.scene.camera.unproject(
                x, y, depth,
                self.widget.frame.width,
                self.widget.frame.height
            )

            self.last_point = world

        self.widget.scene.scene.render_to_depth_image(depth_callback)

        return getattr(self, "last_point", None)

    # ================= 添加线 =================
    def add_line(self, p1, p2):
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([p1, p2])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

        name = f"line_{len(self.lines)}"

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0

        self.widget.scene.add_geometry(name, line, mat)

        self.lines.append({
            "name": name,
            "p1": p1,
            "p2": p2
        })

    # ================= 主入口 =================
    def run(self):
        gui.Application.instance.initialize()

        # ⭐ 1. 数据准备（替代你之前的所有前处理）
        self.prepare_data()

        # ⭐ 2. GUI + 相机初始化
        self.init_scene()

        gui.Application.instance.run()