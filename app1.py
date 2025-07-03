from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# Setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8 + DeepSORT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Interactive Interface for YOLOv8 + DeepSORT")

# Sidebar
st.sidebar.header("DL Model Config")

# Model options
task_type = st.sidebar.selectbox("Select Task", ["Detection and Tracking"])

# Add model uploader for custom YOLOv8 model
uploaded_model = st.sidebar.file_uploader("Upload Custom YOLOv8 Model (.pt file)", type=["pt"])

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if uploaded_model:
    with open("best.pt", "wb") as f:
        f.write(uploaded_model.read())
    model_path = Path("best.pt")
else:
    model_type = st.sidebar.selectbox("Select Model", config.DETECTION_MODEL_LIST)
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    else:
        st.error("Please Select Model in Sidebar")

# Load pretrained model and DeepSORT tracker
try:
    model = load_model(model_path)
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)  # Initialize DeepSORT tracker
except Exception as e:
    st.error(f"Error loading model or tracker: {e}")

# Function to convert YOLO detections to DeepSORT format
def yolo_to_deepsort_format(detections):
    converted_detections = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0]  # YOLO bounding box
        confidence = det.confidence  # Detection confidence
        class_id = det.class_id  # Class ID
        converted_detections.append([x1, y1, x2, y2, confidence, class_id])
    return converted_detections

# Function to draw unique IDs on the frame
def draw_detections_with_ids(image, tracks):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x1, y1, x2, y2 = track.to_tlbr()  # Bounding box
        track_id = track.track_id  # Unique ID
        label_text = f"ID: {track_id}"  # Replace class with unique ID

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)

        # Draw text background and label
        text_size = draw.textsize(label_text, font=font)
        draw.rectangle([x1, y1 - 25, x1 + text_size[0] + 4, y1], fill="blue")
        draw.text((x1 + 2, y1 - 22), label_text, fill="white", font=font)
    return image

# Sidebar for selecting input source
st.sidebar.header("Input Source")
source_selectbox = st.sidebar.selectbox("Select Source", config.SOURCES_LIST)

if source_selectbox == config.SOURCES_LIST[0]:  # Image
    image = infer_uploaded_image(confidence, model)
    if image:
        detections = yolo_to_deepsort_format(image.detections)
        tracks = tracker.update_tracks(detections, image)  # Update DeepSORT
        annotated_image = draw_detections_with_ids(image, tracks)
        st.image(annotated_image, caption="Tracked Image with IDs")

elif source_selectbox == config.SOURCES_LIST[1]:  # Video
    video_frames = infer_uploaded_video(confidence, model)
    if video_frames:
        for frame in video_frames:
            frame_pil = Image.fromarray(frame)
            detections = yolo_to_deepsort_format(frame.detections)
            tracks = tracker.update_tracks(detections, frame)  # Update DeepSORT
            annotated_frame = draw_detections_with_ids(frame_pil, tracks)
            st.image(annotated_frame, caption="Tracked Video Frame with IDs")

elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
    webcam_frames = infer_uploaded_webcam(confidence, model)
    if webcam_frames:
        for frame in webcam_frames:
            frame_pil = Image.fromarray(frame)
            detections = yolo_to_deepsort_format(frame.detections)
            tracks = tracker.update_tracks(detections, frame)  # Update DeepSORT
            annotated_frame = draw_detections_with_ids(frame_pil, tracks)
            st.image(annotated_frame, caption="Tracked Webcam Frame with IDs")

else:
    st.error("Unsupported source selected.")