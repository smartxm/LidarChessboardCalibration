import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import csv
from datetime import datetime


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
        self.corner_sphere_radius = 0.01
        self.selected_corner_idx = None
        self.dragging_corner = False

        self.mouse_origin = None
        self.corner_origin = None
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

        # 监听鼠标操作和键盘操作
        scene.set_on_mouse(self.on_mouse)
        scene.set_on_key(self.on_key)

        return window, scene
    

    def pick_point(self, x, y):
        """
        把在屏幕上点的一个像素(x,y)，转换成三维空间中的一个点
        """
        # 获取当前相机参数（从相机矩阵反推，而不是用初始化保存的参数，事实证明这样更稳定）
        camera = self.scene.scene.camera
        # 视图矩阵：把“世界坐标”变成“相机坐标”（作用：不动相机，而是把整个世界“变换”到相机前面）内部包含：1.相机位置（在哪） 2. 相机朝向（看哪）
        view_matrix = camera.get_view_matrix()
        # 投影矩阵：把3D点投影到2D屏幕  内部包含：1.视场角（FOV）2.宽高比（aspect）3.近裁剪面 / 远裁剪面
        proj_matrix = camera.get_projection_matrix()
        
        # 从视图矩阵获取相机位置（视图矩阵的逆的平移部分）
        view_matrix_np = np.array(view_matrix)
        proj_matrix_np = np.array(proj_matrix)
        
        # 获取视图矩阵的逆，从中提取相机位置和方向
        view_inv = np.linalg.inv(view_matrix_np)
        eye = view_inv[:3, 3]  # 相机位置
        """
        一个标准的4*4变换矩阵
        | R11 R12 R13 Tx |
        | R21 R22 R23 Ty |
        | R31 R32 R33 Tz |
        |  0   0   0  1  |
        这里取的是[Tx, Ty, Tz],即相机在世界坐标中的位置(x, y, z)"""
        forward = -view_inv[:3, 2]  # 相机朝向（OpenGL 约定相机的朝向是 -Z（z粥负方向））
        forward /= np.linalg.norm(forward)
        
        up = view_inv[:3, 1]  # 相机的上方向
        up /= np.linalg.norm(up)
        
        # 相机坐标系
        # 叉乘建立一个一个同时垂直于 forward 和 up 的向量
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        
        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)

        # 屏幕坐标转 NDC
        w = self.scene.frame.width
        h = self.scene.frame.height
        # 把屏幕坐标变成：[-1, 1] 范围
        nx = (x / w - 0.5) * 2      # [-1,1]
        ny = (0.5 - y / h) * 2      # [-1,1], 上为正    屏幕 y：向下是正 数学坐标：向上是正

        # 从投影矩阵获取 FOV 和 aspect
        # 投影矩阵中可以提取出 FOV 信息（从投影矩阵“反推出”视场角）
        # proj_matrix[0,0] = cot(fov_x/2) / aspect, proj_matrix[1,1] = cot(fov_y/2)
        fovy = 2.0 * np.arctan(1.0 / proj_matrix_np[1, 1])
        aspect = w / h
        
        # 把屏幕坐标映射到“相机空间”
        px = nx * np.tan(fovy / 2) * aspect
        py = ny * np.tan(fovy / 2)

        # 射线方向（相机空间 → 世界空间）
        # 射线 = 朝前 + 横向偏移 + 竖向偏移
        ray_dir = forward + px * right + py * true_up
        ray_dir /= np.linalg.norm(ray_dir)
        # 射线起点从相机发射
        ray_origin = eye

        # 射线和平面求交
        plane_center = self.plane.center
        normal = self.plane.R[:, 2]

        denom = np.dot(ray_dir, normal)
        if abs(denom) < 1e-6:
            return None  # 射线平行于平面

        t = np.dot(plane_center - ray_origin, normal) / denom
        if t < 0:
            return None  # 射线指向平面背面
        
        # 返回射线与平面的交点
        hit = ray_origin + t * ray_dir
        return hit

    def on_mouse(self, event):
        """统一管理所有的鼠标操作与更新调用"""
        # 如果在选点模式，则监听
        if self.pick_mode:
            # 按下了 Shift + 鼠标左键
            if (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
                (event.buttons & int(gui.MouseButton.LEFT)) and
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
                    self.update_corner_spheres()
                    self.update_grid_by_corners()

                # 这一行的作用是告诉open3d 这个事件我已经处理完了
                return gui.Widget.EventCallbackResult.HANDLED
            # 这一行的作用是告诉open3d 这个事件我没有处理
            return gui.Widget.EventCallbackResult.IGNORED

        # 拖拽模式：拖动已经选好的 4 个角点
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and (event.buttons & int(gui.MouseButton.LEFT)):
            # 检测点击了哪个角点
            ray_origin, ray_dir = self.get_camera_ray(event.x, event.y)
            idx = self.detect_corner_hit(ray_origin, ray_dir)
            # 检测到鼠标按住的点
            if idx is not None:
                self.selected_corner_idx = idx
                self.dragging_corner = True         # 当前正在拖拽某个角点
                # 记录拖拽起点
                self.mouse_origin = np.array([event.x, event.y])
                self.corner_origin = self.corner_points[idx].copy()
                self.update_corner_spheres()
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.IGNORED

        # 执行拖动
        if event.type == gui.MouseEvent.Type.DRAG and self.dragging_corner and self.selected_corner_idx is not None:
            if self.mouse_origin is not None and self.corner_origin is not None:
                # 计算位移
                displacement = self.move_corner(self.mouse_origin, np.array([event.x, event.y]))
                if displacement is not None:
                    # 更新角点位置
                    self.corner_points[self.selected_corner_idx] = self.corner_origin + displacement
                    # 刷新显示
                    self.update_corner_spheres()
                    self.update_grid_by_corners()
            return gui.Widget.EventCallbackResult.CONSUMED

        # 鼠标松开，结束拖拽状态，清空状态
        if (event.type == gui.MouseEvent.Type.BUTTON_UP and
            (event.buttons & int(gui.MouseButton.LEFT)) and
            self.dragging_corner):
            self.dragging_corner = False
            self.selected_corner_idx = None
            self.mouse_origin = None
            self.corner_origin = None
            self.update_corner_spheres()
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED
    
    def add_pick_point(self, pos):
        """在你点击的位置创建一个蓝色小球"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.corner_sphere_radius)
        # 将创建出来的球移动到目标位置（新建出来的球默认默认位置在0,0,0）
        sphere.translate(pos)
        # 设置球体颜色
        sphere.paint_uniform_color([0, 0, 1])

        name = f"corner_{len(self.corner_points) - 1}"

        # 创建材质
        mat = rendering.MaterialRecord()
        # 设置 Shader，GPU渲染方式，不受光照影响
        mat.shader = "defaultUnlit"

        # 添加到场景（最终显示在gui图形界面）
        self.scene.scene.add_geometry(name, sphere, mat)    

    def get_camera_ray(self, x, y):
        """与pick_point函数算法基本相同，获取相机参数，方便后续调用，负责从相机发射射线"""
        camera = self.scene.scene.camera
        view_matrix = np.array(camera.get_view_matrix())
        proj_matrix = np.array(camera.get_projection_matrix())

        inv_view = np.linalg.inv(view_matrix)
        eye = inv_view[:3, 3]

        forward = -inv_view[:3, 2]
        forward /= np.linalg.norm(forward)

        up = inv_view[:3, 1]
        up /= np.linalg.norm(up)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)

        w = self.scene.frame.width
        h = self.scene.frame.height
        nx = (x / w - 0.5) * 2
        ny = (0.5 - y / h) * 2

        fovy = 2.0 * np.arctan(1.0 / proj_matrix[1, 1])
        aspect = w / h

        px = nx * np.tan(fovy / 2) * aspect
        py = ny * np.tan(fovy / 2)

        ray_dir = forward + px * right + py * true_up
        ray_dir /= np.linalg.norm(ray_dir)

        return eye, ray_dir

    def intersect_ray_plane(self, origin, direction, plane_point, normal):
        """计算一条射线和一个平面的交点，返回交点  
        origin：射线起点；  
        direction：射线方向；  
        plane_point：平面上的任意一个点；  
        normal：平面法向量"""
        # 先将射线方向向量和法向量点成，如果点乘为0则二者垂直，无法求出交点
        denom = np.dot(direction, normal)
        # 浮点数计算不精确，不能直接判断==0
        if abs(denom) < 1e-6:
            return None

        # 射线和平面求交公式
        t = np.dot(plane_point - origin, normal) / denom
        if t < 0:
            return None

        return origin + t * direction

    def ray_sphere_intersection(self, origin, direction, center, radius):
        """射线检测，判断鼠标射线有没有打中角点小球  
        center：球心位置"""
        oc = origin - center
        # 联立射线方程和球面方程 
        # P(t) = origin + t * direction
        # ||P - center||^2 = r^2
        # 化简得(oc + t * direction) ⋅ (oc + t * direction) = r^2
        # 展开，移项t^2(direction⋅direction)+2t(direction⋅oc)+(oc⋅oc−r^2)=0
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(direction, oc)
        c = np.dot(oc, oc) - radius * radius

        delta = b * b - 4.0 * a * c
        if delta < 0:
            return False, None

        # 求根公式
        t1 = (-b - np.sqrt(delta)) / (2.0 * a)
        t2 = (-b + np.sqrt(delta)) / (2.0 * a)
        if t1 >= 0:
            return True, t1
        if t2 >= 0:
            return True, t2
        return False, None

    def detect_corner_hit(self, origin, direction):
        """根据射线检测射线指向的具体是哪一个点（返回索引）"""
        hit_idx = None
        # 初始化最小距离（正无穷）
        min_t = float('inf')
        # 遍历角点，使用enumerate可以同时拿到索引与点的信息
        for i, pos in enumerate(self.corner_points):
            hit, t = self.ray_sphere_intersection(origin, direction, pos, self.corner_sphere_radius)
            if hit and t is not None and t < min_t:
                min_t = t
                hit_idx = i
        return hit_idx

    def move_corner(self, mouse_origin, mouse_current):
        """将鼠标移动了多少转换成平面上的3D位移，返回位移向量"""
        # 获取“起始鼠标”的射线
        ray_origin, ray_dir_origin = self.get_camera_ray(mouse_origin[0], mouse_origin[1])
        # 获取“当前鼠标”的射线（相机位置不变，不需要重复获取）
        _, ray_dir_current = self.get_camera_ray(mouse_current[0], mouse_current[1])

        # 定义拖拽平面（获取点和法向量）
        plane_point = self.plane.center
        normal = self.plane.R[:, 2]

        start_hit = self.intersect_ray_plane(ray_origin, ray_dir_origin, plane_point, normal)
        current_hit = self.intersect_ray_plane(ray_origin, ray_dir_current, plane_point, normal)
        if start_hit is None or current_hit is None:
            return None

        # 最终返回位移（位移向量）
        return current_hit - start_hit

    def update_corner_spheres(self):
        """刷新显示，重新绘制所有角点球"""
        # 删掉旧球 → 创建新球 → 重新显示
        for i, pos in enumerate(self.corner_points):
            name = f"corner_{i}"
            # 删除旧球
            self.scene.scene.remove_geometry(name)
            # 重新创建球并更新位置
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.corner_sphere_radius)
            sphere.translate(pos)

            # 正在拖拽：渲染成绿色
            if i == self.selected_corner_idx and self.dragging_corner:
                sphere.paint_uniform_color([0, 1, 0])
            # 选中了但没拖：渲染成黄色
            elif i == self.selected_corner_idx:
                sphere.paint_uniform_color([1, 1, 0])
            # 普通状态：蓝色
            else:
                sphere.paint_uniform_color([0, 0, 1])

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
        # 10x7 角点对应 9 条竖线和 6 条横线（内部线）
        x_lines = np.linspace(0, x_len, 9)  # 9 条竖线
        y_lines = np.linspace(0, y_len, 6)   # 6 条横线

        points_3d = []
        lines = []
        colors = []

        # 法向：使用当前相机方向判断符号
        normal = self.plane.R[:, 2]
        eye, _ = self.get_camera_ray(self.scene.frame.width * 0.5, self.scene.frame.height * 0.5)
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

        # 删除旧 grid（安全版）
        if self.scene.scene.has_geometry("grid"):
            self.scene.scene.remove_geometry("grid")

        # 添加新 grid
        self.scene.scene.add_geometry("grid", line_set, mat)

        # 保存线的信息
        self.x_lines = x_lines
        self.y_lines = y_lines
        self.grid_origin = p0
        self.grid_x_axis = x_axis
        self.grid_y_axis = y_axis
    def on_key(self, event):

        if event.type == gui.KeyEvent.Type.DOWN:

            # 按下 S 键导出
            if event.key == gui.KeyName.S:

                print("⌨️ S detected → 导出网格点")

                # 生成当前时间字符串
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

                # 文件名
                filename = f"../output/lidar_corner_{current_time}.csv"

                # 导出
                self.export_grid_intersections(filename)

                print(f"✅ 已导出: {filename}")

                return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def export_grid_intersections(self, filename="grid_points.csv"):
        """网格求交，按照固定顺序导出成csv文件"""
        # 检查网格是否存在 hasattr():“has attribute对象有没有这个属性”
        if not hasattr(self, "x_lines"):
            print("❌ 网格还没生成")
            return

        points = []

        p0 = self.grid_origin
        x_axis = self.grid_x_axis
        y_axis = self.grid_y_axis

        # 依旧获取平面法向量
        normal = self.plane.R[:, 2]

        # ✅ 行优先：从 p0 开始，一行一行扫
        # 外层循环：行，内层循环：列
        for row_idx, y in enumerate(self.y_lines):
            for col_idx, x in enumerate(self.x_lines):

                # 将局部坐标 转换成 世界坐标
                pt = p0 + x * x_axis + y * y_axis
                # pt = pt + self.offset * normal  #这里不能再有偏置了，需要输出到文件

                points.append(pt)

                # 调试输出
                print(f"输出：row={row_idx}, col={col_idx}, pt={pt}")

        points = np.array(points)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(points)

        print(f"✅ 已导出 {len(points)} 个点到 {filename}")

    # ================= gui界面主入口 =================
    def run(self):
        app = gui.Application.instance
        app.initialize()

        self.ProjectToPlane()
        # 初始化场景
        window, self.scene = self.init_scene()

        # 运行
        app.run()
