# LidarChessboardCalibration

## 项目简介

这是一个基于Python和Open3D开发的激光雷达标定工具，主要用于激光雷达与相机之间的外参标定。通过交互式GUI界面，用户可以在点云数据中选择棋盘格的四个角点，自动生成网格并导出角点坐标，用于后续的标定计算。

## 功能特性

- **平面检测**: 自动检测点云中的平面区域，支持选择合适的标定平面
- **交互式标定**: 提供直观的3D GUI界面，支持鼠标选择和拖拽角点
- **网格生成**: 根据四个角点自动生成棋盘格网格线，便于精确定位
- **数据导出**: 将角点坐标导出为CSV格式，方便后续标定算法使用
- **实时预览**: 支持实时调整角点位置和网格显示

## 安装要求

### 系统要求
- Python 3.7+
- 支持Open3D可视化的图形界面环境

### 依赖包
```
open3d>=0.16.0
numpy>=1.21.0
```

### 安装步骤
1. 克隆项目到本地：
```bash
git clone https://github.com/smartxm/LidarChessboardCalibration.git
cd LidarChessboardCalibration
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

如果没有requirements.txt文件，请手动安装：
```bash
pip install open3d numpy
```

## 使用方法

### 数据准备
将激光雷达点云数据保存为CSV格式，包含以下列：
- X, Y, Z: 点云坐标
- Reflectivity: 反射率（可选，用于可视化）

示例数据文件放在`input/`目录下。

### 运行标定程序

1. **第一阶段：平面检测和选择**
```bash
cd src
python PlanarPatchDetection.py
```

程序将：
- 加载点云数据
- 检测平面
- 显示平面可视化界面
- 提示用户调整视角后按Q退出

2. **第二阶段：选择标定平面**
在可视化界面中：
- 使用Shift+左键点击选择标定平面上的一点
- 程序自动选择对应的平面并裁剪子点云

3. **第三阶段：GUI标定界面**
程序自动进入标定GUI：
- 使用Shift+左键依次选择棋盘格的4个角点
- 选择完成后，可以拖拽角点进行微调
- 按S键导出角点坐标到`output/`目录

### 输出结果
导出的CSV文件包含棋盘格角点的3D坐标，按行优先顺序排列：
```
x,y,z
x1,y1,z1
x2,y2,z2
...
```

## 文件结构

```
LidarChessboardCalibration/
├── README.md                 # 项目说明文档
├── input/                    # 输入数据目录
│   ├── 1.csv                # 示例点云数据
│   ├── 2.csv
│   └── 3.csv
├── output/                   # 输出结果目录
│   └── lidar_corner_YYYY-MM-DD-HH-MM.csv  # 导出的角点坐标
└── src/                      # 源代码目录
    ├── PlanarPatchDetection.py    # 平面检测和选择模块
    └── lidar_gui_calibrator.py    # GUI标定界面模块
```

## 工作原理

### 1. 平面检测原理
使用Open3D的`detect_planar_patches`方法，通过RANSAC算法检测点云中的平面：
- 计算点云法向量
- 根据法向量方差和共面性阈值筛选平面
- 返回定向包围盒(Oriented Bounding Box)表示检测到的平面

### 2. 点云投影
将选定的子点云投影到检测到的平面上：
- 计算点到平面的距离
- 沿法向量方向投影点云
- 保持颜色信息用于可视化

### 3. 交互式角点选择
- 将鼠标屏幕坐标转换为3D射线
- 计算射线与平面的交点
- 支持拖拽调整角点位置

### 4. 网格生成
根据四个角点建立局部坐标系：
- 计算X、Y方向向量
- 生成9条竖线和6条横线（对应8x5内部网格）
- 计算所有网格交点坐标

### 5. 坐标导出
按行优先顺序导出所有网格交点：
- 确保坐标系一致性
- CSV格式便于后续处理

## 常见问题

**Q: 程序运行时显示"没有选点"**
A: 确保在可视化界面中正确使用Shift+左键选择点。

**Q: 平面检测失败**
A: 检查点云数据质量，调整`detect_planar_patches`的参数。

**Q: GUI界面无法显示**
A: 确保系统支持Open3D可视化，可能需要安装图形界面相关依赖。
