# app.py
from flask import Flask, request, jsonify
import os
import features  # Import the features module containing TensorFlow image/video classification functions
from flask_cors import CORS

app = Flask(__name__)


CORS(app)

@app.route('/')
def home():
    return jsonify(message='API is running')

@app.route('/predictVideo', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify(message="No video part"), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify(message="No selected video"), 400

    # Save the uploaded video to a local path
    video_path = "video.mp4"
    video.save(video_path)

    try:
        # Call the video classifier function from features.py
        prediction = features.video_classifier(video_path)
        return jsonify(result=prediction)
    except Exception as e:
        return jsonify(message=str(e)), 500

# @app.route('/predictImage', methods=['POST'])
# def predict_image():
#     if 'image' not in request.files:
#         return jsonify(message="No image part"), 400

#     image = request.files['image']
#     if image.filename == '':
#         return jsonify(message="No selected image"), 400
#     image_path = 'images/image.jpg'
#     image.save(image_path)

#     try:
#         prediction = features.image_classifier(image_path)
#         return jsonify(result=prediction) 
#     except Exception as e:
#         print(e)
#         return jsonify(message=str(e)), 500  

@app.route('/predictImage', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify(message="No image part"), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify(message="No selected image"), 400

    image_path = 'images/image.jpg'
    image.save(image_path)

    try:
        predicted_class_index = features.image_classifier(image_path)
        
        if predicted_class_index % 2 == 0:
            result = 0  # Real
        else:
            result = 1  # Fake

        return jsonify(result=result)
    except Exception as e:
        print(e)
        return jsonify(message=str(e)), 500

# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
