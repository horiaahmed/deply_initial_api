from flask import Flask, request, jsonify
import cv2
from rtmlib.tools.solution import BodyWithFeet, PoseTracker, Wholebody
import os
import pandas as pd
import joblib
import numpy as np
from sklearn import set_config
import uuid

app = Flask(__name__)

# Set sklearn config
set_config(assume_finite=True)

# Config
device = 'cpu'  # Changed to CPU to rule out CUDA issues
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
def load_models():
    models = {
        "scaler_stage1": joblib.load("scaler_tfb5.joblib"),
        "pca_stage1": joblib.load("pca_tfb5.joblib"),
        "model_stage1": joblib.load("XGBoost_tfb5.joblib"),
        "scaler_stage2": joblib.load("scaler_tfb.joblib"),
        "pca_stage2": joblib.load("pca_tfb.joblib"),
        "model_stage2": joblib.load("XGBoost_tfb.joblib")
    }
    print(f"Stage 1 model classes: {models['model_stage1'].classes_}")
    print(f"Stage 2 model classes: {models['model_stage2'].classes_}")
    return models

models = load_models()

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
    print(f"Original frame count: {original_frame_count}")
    if original_frame_count < 10:
        raise ValueError("Video is too short; need at least 10 frames.")

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
    final_frames = interpolated_frames[:target_frame_count]
    print(f"Interpolated frame count: {len(final_frames)}")
    return final_frames

# Keypoint extraction
def get_keypoints(interpolated_frames):
    all_points = []
    valid_frames = 0
    expected_keypoints = 42  # Number of keypoints per frame
    keypoint_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        92, 95, 96, 98, 100, 102, 104, 106, 108, 110, 112, 113, 116, 117,
                        119, 121, 123, 125, 127, 129, 131, 133]

    for frame_idx, frame in enumerate(interpolated_frames):
        keypoints, scores = body_feet_tracker(frame)
        print(f"Frame {frame_idx + 1}: Keypoints shape: {keypoints.shape}")
        if keypoints.shape[0] == 0:
            print(f"Frame {frame_idx + 1}: No keypoints detected")
            continue
        if keypoints.shape[1] < max(keypoint_indices):
            print(f"Frame {frame_idx + 1}: Expected at least {max(keypoint_indices)} keypoints, got {keypoints.shape[1]}")
            continue
        valid_frames += 1
        frame_points = []
        for p in keypoint_indices:
            try:
                x, y = keypoints[0][p - 1]
                frame_points.extend([float(x), float(y)])
            except IndexError as e:
                print(f"Keypoint index error at frame {frame_idx + 1}, index {p - 1}: {e}")
                return []
        if len(frame_points) != expected_keypoints * 2:
            print(f"Frame {frame_idx + 1}: Expected {expected_keypoints * 2} coordinates, got {len(frame_points)}")
            return []
        all_points.extend(frame_points)

    print(f"Processed {valid_frames} valid frames with {len(all_points)} total coordinates")
    if valid_frames < 100:
        raise ValueError(f"Only {valid_frames} frames had detectable keypoints; need at least 100.")
    if len(all_points) != 10080:
        raise ValueError(f"Extracted {len(all_points)} coordinates; expected 10080.")
    return all_points

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded video temporarily
        video_path = f"temp_{uuid.uuid4()}.mp4"
        video_file.save(video_path)

        target_frame_count = 120
        stage1_classes = {0: "autistic", 1: "normal"}
        stage2_classes = {0: 'pacing', 1: 'toe_walking', 2: 'arm_flapping', 3: 'spinning', 4: 'headbang'}
        expected_keypoints = 42
        expected_features = 120 * expected_keypoints * 2  # 10080

        # Process video
        interpolated_frames = interpolate_frames(video_path, target_frame_count)
        all_points = get_keypoints(interpolated_frames)

        if len(all_points) != expected_features:
            os.remove(video_path)
            return jsonify({
                'error': f"Extracted {len(all_points)} features; expected {expected_features}",
                'suggestion': 'Ensure the video is clear, well-lit, and shows a person in the frame'
            }), 400

        # Perform predictions
        all_points = np.array(all_points).reshape(1, -1)
        df_all_points = pd.DataFrame(all_points, columns=[f'F{i}' for i in range(1, expected_features + 1)])

        # Stage 1 prediction
        scaled_stage1 = models["scaler_stage1"].transform(df_all_points)
        pca_stage1_data = models["pca_stage1"].transform(scaled_stage1)
        stage1_pred = models["model_stage1"].predict(pca_stage1_data)[0]
        print(f"Stage 1 prediction: {stage1_pred}")
        if stage1_pred not in stage1_classes:
            raise ValueError(f"Invalid Stage 1 prediction: {stage1_pred}")
        stage1_label = stage1_classes[stage1_pred]

        # Stage 2 prediction if autistic
        stage2_label = None
        if stage1_label == "autistic":
            scaled_stage2 = models["scaler_stage2"].transform(df_all_points)
            pca_stage2_data = models["pca_stage2"].transform(scaled_stage2)
            stage2_pred = models["model_stage2"].predict(pca_stage2_data)[0]
            print(f"Stage 2 prediction: {stage2_pred}")
            if stage2_pred not in stage2_classes:
                raise ValueError(f"Invalid Stage 2 prediction: {stage2_pred}")
            stage2_label = stage2_classes[stage2_pred]

        # Clean up
        os.remove(video_path)

        # Prepare response
        response = {
            'stage1_prediction': stage1_label,
            'stage2_behavior': stage2_label,
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        # Clean up in case of error
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
