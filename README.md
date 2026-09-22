# RoadVision 🚗

**基于 YOLO + ByteTrack 的道路车辆检测、跟踪与交通流分析系统**

RoadVision 是一个端到端的计算机视觉项目，用于对道路监控视频中的车辆进行自动检测、多目标跟踪、虚拟检测线计数和交通流量统计分析。系统使用 YOLO 预训练模型进行目标检测，使用 ByteTrack 算法进行多目标跟踪并分配稳定 ID，支持按车辆类型和行驶方向分类计数，并计算交通流量（veh/h）。

---

## 📋 功能介绍

- **车辆目标检测**：使用 YOLO 预训练模型，支持 car、truck、bus、motorcycle 四类车辆
- **多目标跟踪**：基于 ByteTrack 算法，为每辆车分配稳定的跟踪 ID
- **运动轨迹绘制**：在画面上实时绘制每辆车的历史运动轨迹
- **虚拟检测线计数**：设置一条虚拟检测线，车辆穿过时自动计数
- **行驶方向判断**：自动判断车辆穿过检测线的方向（上/下 或 左/右）
- **分类统计**：按车辆类型和方向分别计数
- **交通流量计算**：根据时间窗口计算交通流量（veh/h）
- **可视化输出**：在画面上绘制检测框、类别、置信度、车辆 ID 和运动轨迹
- **结果导出**：输出标注后的视频和 CSV 统计文件
- **Web 界面**：提供 Streamlit Web 页面，支持视频上传和参数配置

---

## 🛠 技术栈

| 组件 | 说明 |
|------|------|
| [YOLO (Ultralytics)](https://github.com/ultralytics/ultralytics) | 目标检测，使用预训练模型（如 YOLO11n） |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | 多目标跟踪算法，内置于 Ultralytics |
| [OpenCV](https://opencv.org/) | 视频读写、图像绘制 |
| [Streamlit](https://streamlit.io/) | Web 界面 |
| [PyTorch](https://pytorch.org/) | 深度学习后端 |
| Python | 编程语言（>=3.8） |

---

## 🔄 系统流程图

```
┌─────────────┐
│  输入视频    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  YOLO 车辆检测       │  ← 预训练模型 (yolo11n.pt)
│  (car/truck/bus/mc)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  ByteTrack 多目标跟踪 │  ← 稳定 ID 分配
│  (轨迹历史管理)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  虚拟检测线计数       │  ← 穿线检测 + 方向判断
│  (分类: 类型 × 方向)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  交通流量分析         │  ← veh/h 计算
│  (时间窗口统计)       │
└──────┬──────────────┘
       │
       ▼
┌──────────┐  ┌──────────┐
│ 标注视频  │  │ CSV 统计 │
└──────────┘  └──────────┘
```

---

## 📁 项目结构

```
RoadVision/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖列表
├── .gitignore                # Git 忽略规则
├── app.py                    # Streamlit Web 应用入口
├── configs/
│   └── config.yaml           # 全局配置文件（模型、阈值、检测线等）
├── src/
│   ├── __init__.py           # 模块导出
│   ├── detector.py           # YOLO 车辆检测模块
│   ├── tracker.py            # ByteTrack 轨迹管理模块
│   ├── counter.py            # 虚拟检测线计数模块
│   ├── traffic_analyzer.py   # 交通流量分析模块
│   ├── visualizer.py         # 可视化绘制模块
│   └── pipeline.py          # 端到端处理流水线
├── scripts/
│   └── run_video.py          # 命令行批处理脚本
├── assets/                   # 测试视频存放目录
├── outputs/                  # 输出结果目录（标注视频 + CSV）
└── ultralytics/              # YOLO 核心库（Ultralytics）
```

### 模块职责说明

| 模块 | 职责 |
|------|------|
| `src/detector.py` | 加载 YOLO 预训练模型，对每帧执行检测+跟踪，返回 `Detection` 列表 |
| `src/tracker.py` | 管理每辆车的轨迹历史（中心点队列），供轨迹绘制和穿线检测使用 |
| `src/counter.py` | 通过比较前后帧位置检测虚拟线穿越，判断方向，按类型和方向分类计数 |
| `src/traffic_analyzer.py` | 按时间窗口计算 veh/h 流量，导出 CSV 统计文件 |
| `src/visualizer.py` | 在帧上绘制检测框、类别、置信度、ID、轨迹、检测线和统计 HUD |
| `src/pipeline.py` | 编排以上模块，完成端到端视频处理 |

---

## 📥 安装方法

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/RoadVision.git
cd RoadVision
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **说明**：`ultralytics` 会自动安装 PyTorch、OpenCV 等依赖。首次运行时 YOLO 模型权重（如 `yolo11n.pt`）会自动从官方源下载。

---

## 🚀 运行方法

### 方式一：命令行运行

```bash
python scripts/run_video.py --video path/to/road_video.mp4
```

**常用参数：**

```bash
# 指定模型和置信度
python scripts/run_video.py --video road.mp4 --model yolov8s.pt --conf 0.4

# 设置垂直检测线（车辆从左向右或从右向左行驶）
python scripts/run_video.py --video road.mp4 --line-orientation vertical --line-position 0.3

# 使用 CPU 推理
python scripts/run_video.py --video road.mp4 --device cpu

# 指定输出目录
python scripts/run_video.py --video road.mp4 --output-dir my_results
```

**所有参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | （必填） | 输入视频路径 |
| `--config` | `configs/config.yaml` | 配置文件路径 |
| `--model` | `yolo11n.pt` | 模型名称/路径 |
| `--conf` | `0.3` | 置信度阈值 |
| `--device` | `auto` | 推理设备 (auto/cpu/0) |
| `--line-orientation` | `horizontal` | 检测线方向 (horizontal/vertical) |
| `--line-position` | `0.5` | 检测线位置 (0.0-1.0) |
| `--output-dir` | `outputs` | 输出目录 |
| `--no-save-video` | `False` | 不保存标注视频 |
| `--no-save-csv` | `False` | 不保存 CSV |

### 方式二：Streamlit Web 界面

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，在界面上：
1. 上传道路视频
2. 在左侧边栏配置模型、置信度、检测线参数
3. 点击"开始分析"按钮
4. 查看标注视频和统计结果
5. 下载处理后的视频和 CSV 文件

---

## 🎬 Demo 使用方法

1. 将一段道路监控视频放入 `assets/` 目录
2. 运行命令行或 Streamlit 进行处理
3. 处理结果保存在 `outputs/` 目录

**推荐测试视频来源：**
- 行车记录仪视频
- 道路监控摄像头视频
- 公开的交通视频数据集（如 [UA-DETRAC](http://detrac-db.rit.albany.edu/)）

> 如果没有现成视频，可以使用任意包含道路上行驶车辆的视频文件进行测试。

<!-- 示例截图占位 -->
<!-- ![demo](assets/demo_placeholder.png) -->
<!-- ![result](assets/result_placeholder.png) -->

---

## 🧠 核心算法说明

### 1. YOLO 目标检测

YOLO（You Only Look Once）是一种单阶段目标检测算法。它将输入图像划分为网格，每个网格直接预测边界框和类别概率。RoadVision 使用 Ultralytics 提供的预训练 YOLO 模型（在 COCO 数据集上训练），只保留车辆相关类别（car=2, motorcycle=3, bus=5, truck=7）的检测结果。

### 2. ByteTrack 多目标跟踪

ByteTrack 是一种简单高效的多人跟踪算法。它的核心思想是：**即使是低置信度的检测结果也可能是有用的**。ByteTrack 分两阶段进行数据关联：
- 第一阶段：用高置信度检测结果与已有轨迹进行匹配（基于 IoU）
- 第二阶段：用低置信度检测结果与未匹配的轨迹进行匹配

这样能减少因遮挡或模糊导致的轨迹中断。RoadVision 通过 Ultralytics 的 `model.track()` 接口调用 ByteTrack，为每辆车分配一个在整个视频中保持稳定的跟踪 ID。

### 3. 虚拟检测线计数

在画面中设置一条虚拟检测线（水平或垂直）。对每辆跟踪到的车辆，比较其前后帧的中心点位置：
- 如果车辆的中心点从前一帧到当前帧穿过了检测线，则计为一次穿越事件
- 根据穿越方向（上/下 或 左/右）对车辆进行分类
- 每辆车只计一次（基于 track ID 去重）

### 4. 交通流量统计

交通流量（veh/h）的计算方法：

```
流量 = (穿越检测线的车辆总数 / 视频时长(秒)) × 3600
```

系统还支持按时间窗口（默认 60 分钟）计算分时段流量，用于观察流量变化趋势。

---

## 📊 输出结果说明

### 标注视频

输出的视频在每一帧上绘制以下信息：
- **检测框**：不同类别使用不同颜色（car=绿色, motorcycle=橙色, bus=红色, truck=蓝色）
- **标签**：`ID:xx car 0.85`（跟踪ID + 类别名 + 置信度）
- **运动轨迹**：每辆车的历史位置点连线（颜色渐变，越远越淡）
- **检测线**：黄色线标记虚拟计数线位置
- **统计 HUD**：右上角半透明面板显示实时统计（帧号、时间、总数、分类计数）

### CSV 统计文件

每行记录一次穿线事件：

| track_id | class | direction | frame_idx | timestamp_s |
|----------|-------|-----------|-----------|-------------|
| 1        | car   | down      | 142       | 4.733       |
| 2        | truck | down      | 185       | 6.167       |
| 3        | car   | up        | 203       | 6.767       |

---

## 🔮 后续可扩展方向

- [ ] 支持实时视频流（RTSP/摄像头）输入
- [ ] 添加车辆速度估计（基于像素位移和标定参数）
- [ ] 支持多条虚拟检测线
- [ ] 增加行人/非机动车检测类别
- [ ] 集成区域热力图分析
- [ ] 支持 GPU 加速推理（TensorRT 导出）
- [ ] 添加 Web 端实时统计看板
- [ ] 支持自定义模型微调训练

---

## 📄 开源协议

本项目基于 [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics) 协议。
