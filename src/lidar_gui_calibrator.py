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

        scene.set_on_mouse(self.on_mouse)
        self.scene = scene

        return window, scene

    def pick_point(self, x, y):
        """
        屏幕点击 → 射线打平面 → 找最近点
        """

        # ===== 相机参数 =====
        center, eye, up = self.cam_parmams
        eye = np.array(eye)
        center = np.array(center)
        up = np.array(up)

        # 相机坐标系
        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        true_up = np.cross(right, forward)

        # ===== 屏幕 → NDC =====
        w = self.scene.frame.width
        h = self.scene.frame.height

        nx = (x / w - 0.5) * 2
        ny = (0.5 - y / h) * 2

        fov = np.deg2rad(60.0)
        aspect = w / h

        px = nx * np.tan(fov / 2) * aspect
        py = ny * np.tan(fov / 2)

        # ===== 射线 =====
        ray_dir = forward + px * right + py * true_up
        ray_dir /= np.linalg.norm(ray_dir)

        ray_origin = eye

        # ===== 与平面求交 =====
        plane_center = self.plane.center
        normal = self.plane.R[:, 2]

        denom = np.dot(ray_dir, normal)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(plane_center - ray_origin, normal) / denom
        if t < 0:
            return None

        hit = ray_origin + t * ray_dir

        # ===== ⭐ 找最近点（关键！）=====
        pts = np.asarray(self.pcd.points)

        dists = np.linalg.norm(pts - hit, axis=1)
        idx = np.argmin(dists)

        nearest = pts[idx]

        return nearest
    
    def on_mouse(self, event):

        if not self.pick_mode:
            return gui.Widget.EventCallbackResult.IGNORED

        # 只响应：Shift + 左键抬起
        if (event.type == gui.MouseEvent.Type.BUTTON_UP and
            event.is_modifier_down(gui.KeyModifier.SHIFT)):

            x = event.x
            y = event.y

            # ⭐ 用深度 picking（替换你之前的 screen_to_plane）
            world = self.pick_point(x, y)

            if world is None:
                print("没点到有效位置")
                return gui.Widget.EventCallbackResult.HANDLED

            print("选中点:", world)

            # 记录
            self.corner_points.append(np.array(world))

            # 显示蓝点
            self.add_pick_point(world)

            # 满4个点 → 生成网格
            if len(self.corner_points) == 4:
                print("4个点选完，生成网格")
                self.pick_mode = False
                self.update_grid_by_corners()

        return gui.Widget.EventCallbackResult.HANDLED
    
    def add_pick_point(self, pos):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(pos)
        sphere.paint_uniform_color([0, 0, 1])

        name = f"pick_{len(self.corner_points)}"

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"

        self.scene.scene.add_geometry(name, sphere, mat)    

    def update_grid_by_corners(self):

        p0, p1, p2, p3 = self.corner_points

        # 两个方向（按点击顺序，不做排序）
        x_vec = (p1 - p0) / 9.0
        y_vec = (p2 - p0) / 6.0

        points_3d = []
        lines = []
        colors = []

        def add_line(p1, p2):
            idx = len(points_3d)
            points_3d.append(p1)
            points_3d.append(p2)
            lines.append([idx, idx + 1])
            colors.append([1, 0, 0])

        # ===== 法向偏移（保持你原来的逻辑）=====
        normal = self.plane.R[:, 2]
        eye = np.asarray(self.cam_parmams[1])
        view_dir = eye - self.plane.center
        view_dir /= np.linalg.norm(view_dir)

        if np.dot(normal, view_dir) < 0:
            normal = -normal

        def offset(p):
            return p + self.offset * normal

        # 竖线
        for i in range(10):
            start = p0 + i * x_vec
            end = start + 6 * y_vec
            add_line(offset(start), offset(end))

        # 横线
        for j in range(7):
            start = p0 + j * y_vec
            end = start + 9 * x_vec
            add_line(offset(start), offset(end))

        # ===== 更新 =====
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 2.0

        self.scene.scene.remove_geometry("grid")
        self.scene.scene.add_geometry("grid", line_set, mat)

        # 删除选点显示
        for i in range(1, 5):
            self.scene.scene.remove_geometry(f"pick_{i}")    

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