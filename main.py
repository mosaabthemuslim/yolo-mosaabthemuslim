import cv2
import streamlit as sl
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
import tempfile

FILE = Path(__file__).resolve()
ROOT = FILE.parent

if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

SOURCES_LIST = ["Image", "Video"]

IMAGES_DIR = ROOT.joinpath("images")

# Load YOLO models
yolo = YOLO("yolo11n.pt")
yolo_pose = YOLO("yolo11n-pose.pt")
yolo_segmentation = YOLO("yolo11n-seg.pt")

# Streamlit configuration
sl.set_page_config(page_title="YOLOv11", layout="wide")
sl.header("You Only Look Once")
sl.sidebar.header("Model Configuration")

# Sidebar controls
model_type = sl.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])
confidence_value = sl.sidebar.slider("Model Confidence", min_value=25, max_value=100, step=10)
source_type = sl.sidebar.selectbox("Input Source", SOURCES_LIST)

# Input handling based on source
if source_type == "Image":
    uploaded_file = sl.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        sl.image(image, caption="Uploaded Image", use_container_width=True)

        # Run the selected model
        if model_type == "Detection":
            results = yolo(source=image, conf=confidence_value / 100)
        elif model_type == "Segmentation":
            results = yolo_segmentation(source=image, conf=confidence_value / 100)
        elif model_type == "Pose Estimation":
            results = yolo_pose(source=image, conf=confidence_value / 100)

        # Display results
        sl.image(results[0].plot(), caption="Model Output", use_container_width=True)

elif source_type == "Video":
    uploaded_video = sl.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Save video to a temporary directory
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Load video using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        st_frame = sl.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run the selected model
            if model_type == "Detection":
                results = yolo(source=frame, conf=confidence_value / 100)
            elif model_type == "Segmentation":
                results = yolo_segmentation(source=frame, conf=confidence_value / 100)
            elif model_type == "Pose Estimation":
                results = yolo_pose(source=frame, conf=confidence_value / 100)

            # Display video frames with results
            annotated_frame = results[0].plot()
            st_frame.image(annotated_frame, channels="BGR")

        cap.release()