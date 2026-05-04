import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np


class LidarGuiCalibrator:
    def __init__(self, sub_pcd, plane, cam_parmams):
        self.original_pcd = sub_pcd
        self.pcd = None
        self.scene = None

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
        self.scene = scene
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

        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 1.0

        scene.set_on_mouse(self.on_mouse)
        

        return window, scene
    

    def pick_point(self, x, y):
        """
        屏幕点击 → 射线打平面 → 找最近点
        返回吸附到投影点云的最近点（世界坐标）
        """
        # 获取当前相机参数（从相机矩阵反推，而不是用初始化保存的参数）
        camera = self.scene.scene.camera
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        
        # 从视图矩阵获取相机位置（视图矩阵的逆的平移部分）
        view_matrix_np = np.array(view_matrix)
        proj_matrix_np = np.array(proj_matrix)
        
        # 获取视图矩阵的逆，从中提取相机位置和方向
        view_inv = np.linalg.inv(view_matrix_np)
        eye = view_inv[:3, 3]  # 相机位置
        forward = -view_inv[:3, 2]  # 相机朝向（OpenGL 约定是 -Z）
        forward /= np.linalg.norm(forward)
        
        up = view_inv[:3, 1]  # 相机的上方向
        up /= np.linalg.norm(up)
        
        # 相机坐标系
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        
        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)

        # 2️⃣ 屏幕坐标转 NDC
        w = self.scene.frame.width
        h = self.scene.frame.height
        nx = (x / w - 0.5) * 2      # [-1,1]
        ny = (0.5 - y / h) * 2      # [-1,1], 上为正

        # 3️⃣ 从投影矩阵获取 FOV 和 aspect
        # 投影矩阵中可以提取出 FOV 信息
        # proj_matrix[0,0] = cot(fov_x/2) / aspect, proj_matrix[1,1] = cot(fov_y/2)
        fovy = 2.0 * np.arctan(1.0 / proj_matrix_np[1, 1])
        aspect = w / h
        
        px = nx * np.tan(fovy / 2) * aspect
        py = ny * np.tan(fovy / 2)

        # 射线方向（相机空间 → 世界空间）
        ray_dir = forward + px * right + py * true_up
        ray_dir /= np.linalg.norm(ray_dir)
        ray_origin = eye

        # 4️⃣ 与平面求交
        plane_center = self.plane.center
        normal = self.plane.R[:, 2]

        denom = np.dot(ray_dir, normal)
        if abs(denom) < 1e-6:
            return None  # 射线平行于平面

        t = np.dot(plane_center - ray_origin, normal) / denom
        if t < 0:
            return None  # 射线指向平面背面

        hit = ray_origin + t * ray_dir

        # 5️⃣ 吸附到最近点（投影后的点云）
        pts = np.asarray(self.pcd.points)
        idx = np.argmin(np.linalg.norm(pts - hit, axis=1))
        nearest = pts[idx]

        return hit

    def on_mouse(self, event):
        if not self.pick_mode:
            return gui.Widget.EventCallbackResult.IGNORED

        if (event.type == gui.MouseEvent.Type.BUTTON_UP and
            event.is_modifier_down(gui.KeyModifier.SHIFT)):

            x, y = event.x, event.y
            world = self.pick_point(x, y)

            if world is None:
                print("没点到有效位置")
                return gui.Widget.EventCallbackResult.HANDLED

            print("选中点:", world)
            self.corner_points.append(np.array(world))
            self.add_pick_point(world)

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

        # 建立局部坐标系
        x_dir = p1 - p0
        y_dir = p2 - p0

        x_len = np.linalg.norm(x_dir)
        y_len = np.linalg.norm(y_dir)

        x_axis = x_dir / x_len
        y_axis = y_dir / y_len

        # 定义局部超出范围，方便后续生成拖拽点不遮挡现有棋盘格
        margin = 0.3

        xmin, xmax = -margin * x_len, (1 + margin) * x_len
        ymin, ymax = -margin * y_len, (1 + margin) * y_len

        # 计算竖线和横线
        # 10x7 角点对应 9 条竖线和 6 条横线
        x_lines = np.linspace(0, x_len, 9)  # 9 条竖线
        y_lines = np.linspace(0, y_len, 6)   # 6 条横线

        points_3d = []
        lines = []
        colors = []

        # 法向
        normal = self.plane.R[:, 2]
        eye = np.asarray(self.cam_parmams[1])
        view_dir = eye - self.plane.center
        view_dir /= np.linalg.norm(view_dir)

        if np.dot(normal, view_dir) < 0:
            normal = -normal

        def to_world(x, y):
            base = p0 + x * x_axis + y * y_axis
            return base + self.offset * normal

        # 竖线添加到场景
        for x in x_lines:
            p1_ = to_world(x, ymin)
            p2_ = to_world(x, ymax)

            idx = len(points_3d)
            points_3d.append(p1_)
            points_3d.append(p2_)
            lines.append([idx, idx + 1])
            colors.append([1, 0, 0])

        # 横线添加到场景
        for y in y_lines:
            p1_ = to_world(xmin, y)
            p2_ = to_world(xmax, y)

            idx = len(points_3d)
            points_3d.append(p1_)
            points_3d.append(p2_)
            lines.append([idx, idx + 1])
            colors.append([1, 0, 0])

        # 更新
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 2.0

        self.scene.scene.remove_geometry("grid")
        self.scene.scene.add_geometry("grid", line_set, mat)

        # 删除选点
        for i in range(1, 5):
            self.scene.scene.remove_geometry(f"pick_{i}") 

    # ================= 主入口 =================
    def run(self):
        app = gui.Application.instance
        app.initialize()

        self.ProjectToPlane()
        # 初始化场景
        window, self.scene = self.init_scene()



        # ===== 4️⃣ 运行 =====
        app.run()