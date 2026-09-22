"""RoadVision Streamlit Web Application.

Run with:
    streamlit run app.py

Provides a simple UI to upload a road video, configure detection/tracking
parameters, run the analysis pipeline, and view the results.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import RoadVisionPipeline, load_config

# --------------------------------------------------------------------------- #
#  Page configuration                                                          #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="RoadVision - Traffic Flow Analysis",
    page_icon="🚗",
    layout="wide",
)

st.title("RoadVision 🚗")
st.markdown("**基于 YOLO + ByteTrack 的道路车辆检测、跟踪与交通流分析系统**")
st.markdown("---")


# --------------------------------------------------------------------------- #
#  Sidebar: configuration                                                      #
# --------------------------------------------------------------------------- #

st.sidebar.header("参数配置")

# Model selection
model_options = {
    "YOLO11n (fastest)": "yolo11n.pt",
    "YOLO11s (balanced)": "yolo11s.pt",
    "YOLOv8n (fast)": "yolov8n.pt",
    "YOLOv8s (balanced)": "yolov8s.pt",
}
model_choice = st.sidebar.selectbox("模型选择", list(model_options.keys()), index=0)
model_path = model_options[model_choice]

# Confidence threshold
conf_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.3, 0.05)

# Device
device = st.sidebar.selectbox("推理设备", ["auto", "cpu", "gpu:0"], index=0)
device_map = {"auto": "", "cpu": "cpu", "gpu:0": "0"}

# Counting line
st.sidebar.subheader("虚拟检测线")
line_orientation = st.sidebar.radio("线的方向", ["horizontal (水平)", "vertical (垂直)"], index=0)
orientation = "horizontal" if "horizontal" in line_orientation else "vertical"
line_position = st.sidebar.slider("线位置 (0.0-1.0)", 0.0, 1.0, 0.5, 0.05)

# Traffic flow window
window_minutes = st.sidebar.number_input("流量统计时间窗口 (分钟)", 1, 120, 60)

# Show trails
show_trails = st.sidebar.checkbox("显示运动轨迹", value=True)
show_hud = st.sidebar.checkbox("显示统计 HUD", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("📁 [项目 GitHub](https://github.com/ultralytics/ultralytics)")


# --------------------------------------------------------------------------- #
#  Main panel                                                                  #
# --------------------------------------------------------------------------- #

# --- Video upload ---
st.subheader("1. 上传视频")
uploaded_file = st.file_uploader(
    "选择一段道路视频 (mp4, avi, mov)",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:
    # Save uploaded file to a temp location
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"视频已上传: {uploaded_file.name}")

    # --- Show original video ---
    st.subheader("2. 原始视频")
    st.video(input_path)

    # --- Run analysis ---
    st.subheader("3. 开始分析")
    run_btn = st.button("🚀 开始分析", type="primary")

    if run_btn:
        # Build config
        config = load_config(os.path.join(project_root, "configs", "config.yaml"))
        config["model"]["path"] = model_path
        config["model"]["device"] = device_map[device]
        config["detection"]["conf_threshold"] = conf_threshold
        config["counting"]["line_orientation"] = orientation
        config["counting"]["line_position"] = line_position
        config["traffic"]["time_window_minutes"] = window_minutes

        # Create pipeline
        with st.spinner("正在加载模型..."):
            try:
                pipeline = RoadVisionPipeline(config)
                # Override visualizer settings
                pipeline.visualizer.show_trails = show_trails
                pipeline.visualizer.show_hud = show_hud
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.stop()

        # Process
        progress_bar = st.progress(0.0, text="正在处理视频...")
        status_text = st.empty()

        def progress_callback(cur, total):
            pct = min(cur / total, 1.0) if total > 0 else 0.0
            progress_bar.progress(pct, text=f"正在处理: {cur}/{total} 帧 ({pct * 100:.1f}%)")

        try:
            result = pipeline.process_video(
                input_path,
                progress_callback=progress_callback,
            )
        except Exception as e:
            st.error(f"处理失败: {e}")
            st.stop()

        progress_bar.progress(1.0, text="处理完成!")
        st.success("视频分析完成!")

        # --- Show results ---
        st.subheader("4. 分析结果")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("总车辆数", result.total_vehicles)
        col2.metric("交通流量", f"{result.overall_rate_per_hour} veh/h")
        col3.metric("视频时长", f"{result.video_duration_s}s")

        # By type
        st.markdown("#### 各车辆类型数量")
        type_cols = st.columns(max(len(result.by_type), 1))
        for i, (cls_name, cnt) in enumerate(sorted(result.by_type.items())):
            type_cols[i].metric(cls_name, cnt)

        # By direction
        st.markdown("#### 各方向车辆数量")
        dir_cols = st.columns(max(len(result.by_direction), 1))
        for i, (direction, cnt) in enumerate(sorted(result.by_direction.items())):
            dir_cols[i].metric(direction, cnt)

        st.markdown("---")

        # Output video
        if result.output_video_path and os.path.exists(result.output_video_path):
            st.subheader("5. 处理后视频")
            st.video(result.output_video_path)

            # Download buttons
            with open(result.output_video_path, "rb") as f:
                st.download_button(
                    "⬇️ 下载处理后的视频",
                    f.read(),
                    file_name=os.path.basename(result.output_video_path),
                    mime="video/mp4",
                )

        if result.output_csv_path and os.path.exists(result.output_csv_path):
            with open(result.output_csv_path, "rb") as f:
                st.download_button(
                    "⬇️ 下载 CSV 统计结果",
                    f.read(),
                    file_name=os.path.basename(result.output_csv_path),
                    mime="text/csv",
                )

else:
    st.info("👆 请在上方上传一段道路视频以开始分析。")
    st.markdown("""
    ### 使用说明

    1. **上传视频** — 选择一段道路监控或行车记录仪视频
    2. **配置参数** — 在左侧边栏选择模型、置信度、检测线位置
    3. **开始分析** — 点击"开始分析"按钮，系统将自动处理
    4. **查看结果** — 处理完成后可查看标注视频和统计数据
    5. **下载结果** — 可下载处理后的视频和 CSV 统计文件

    ### 支持的车辆类别
    - 🚗 Car (汽车)
    - 🏍️ Motorcycle (摩托车)
    - 🚌 Bus (公交车)
    - 🚚 Truck (卡车)
    """)
