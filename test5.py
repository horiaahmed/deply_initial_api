import streamlit as st
import cv2
from rtmlib.tools.solution import BodyWithFeet, PoseTracker, Wholebody
import os
import pandas as pd
import joblib
import numpy as np
from sklearn import set_config

# Set sklearn config
set_config(assume_finite=True)

# Config
device = 'cuda'
backend = 'onnxruntime'
openpose_skeleton = False

# Initialize PoseTracker once
body_feet_tracker = PoseTracker(
    Wholebody,
    det_frequency=7,
    tracking=True,
    to_openpose=openpose_skeleton,
    mode='performance',
    backend=backend,
    device=device
)

# Load models once
@st.cache_resource
def load_models():
    models = {
        "scaler_stage1": joblib.load("scaler_tfb5.joblib"),
        "pca_stage1": joblib.load("pca_tfb5.joblib"),
        "model_stage1": joblib.load("XGBoost_tfb5.joblib"),
        "scaler_stage2": joblib.load("scaler_tfb.joblib"),
        "pca_stage2": joblib.load("pca_tfb.joblib"),
        "model_stage2": joblib.load("XGBoost_tfb.joblib")
    }
    return models

# Interpolation function
def interpolate_frames(video_path, target_frame_count):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    original_frame_count = len(frames)
    if original_frame_count <= 1:
        return frames

    total_frames_needed = target_frame_count - original_frame_count
    num_new_frames = total_frames_needed // (original_frame_count - 1)
    interpolated_frames = []

    for i in range(original_frame_count - 1):
        interpolated_frames.append(frames[i])
        for j in range(num_new_frames + 1):
            alpha = (j + 1) / (num_new_frames + 1)
            new_frame = cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0)
            interpolated_frames.append(new_frame)

    interpolated_frames.append(frames[-1])
    return interpolated_frames[:target_frame_count]

# Keypoint extraction
def get_keypoints(interpolated_frames):
    all_points = []
    for frame in interpolated_frames:
        keypoints, scores = body_feet_tracker(frame)
        if keypoints.shape[0] == 0:
            continue

        for p in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                  92, 95, 96, 98, 100, 102, 104, 106, 108, 110, 112, 113, 116, 117,
                  119, 121, 123, 125, 127, 129, 131, 133]:
            x, y = keypoints[0][p - 1]
            all_points.extend([float(x), float(y)])

    return all_points

# Main Streamlit App
def main():
    st.title("🧠 Autism Behavior Detection from Pose Video")
    st.caption("Using Pose Estimation + Two-stage Machine Learning Pipeline")

    uploaded_file = st.file_uploader("📁 Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(video_path)

        target_frame_count = 120
        models = load_models()

        stage1_classes = {0: "autistic", 1: "normal"}
        stage2_classes = {0: 'pacing', 1: 'toe_walking', 2: 'arm flapping', 3: 'spinning', 4: 'headbang'}

        if st.button("🔍 Analyze Video"):
            with st.spinner("⏳ Processing video and extracting keypoints..."):
                interpolated_frames = interpolate_frames(video_path, target_frame_count)
                all_points = get_keypoints(interpolated_frames)

            expected_features = 10080
            if len(all_points) != expected_features:
                st.error(f"❌ Extracted {len(all_points)} features; expected {expected_features}. Try another video.")
                os.remove(video_path)
                return

            try:
                all_points = np.array(all_points).reshape(1, -1)
                df_all_points = pd.DataFrame(all_points, columns=[f'F{i}' for i in range(1, expected_features + 1)])

                # Stage 1 prediction
                scaled_stage1 = models["scaler_stage1"].transform(df_all_points)
                pca_stage1_data = models["pca_stage1"].transform(scaled_stage1)
                stage1_pred = models["model_stage1"].predict(pca_stage1_data)[0]
                stage1_label = stage1_classes[stage1_pred]

                st.markdown(f"### 🧪 Stage 1 Prediction: **{stage1_label.upper()}**")

                # Stage 2 prediction if autistic
                if stage1_label == "autistic":
                    scaled_stage2 = models["scaler_stage2"].transform(df_all_points)
                    pca_stage2_data = models["pca_stage2"].transform(scaled_stage2)
                    stage2_pred = models["model_stage2"].predict(pca_stage2_data)[0]
                    stage2_label = stage2_classes.get(stage2_pred, "Unknown")

                    st.success(f"🎯 Stage 2 Behavior Detected: **{stage2_label.replace('_', ' ').capitalize()}**")
                else:
                    st.info("No further analysis needed for normal behavior.")

            except Exception as e:
                st.error(f"🔥 Prediction error: {e}")

            os.remove(video_path)

if __name__ == "__main__":
    main()
