from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("Random Forest.joblib")  

def extract_features(video_path):
    return [0.0] * 128  # Replace with real feature extraction

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    filename = os.path.join('uploads', video.filename)
    video.save(filename)

    features = extract_features(filename)
    prediction = model.predict([features])[0]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)