import cv2
import numpy as np
from retinaface import RetinaFace
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, render_template, request, Response, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class FaceDetector:
    def __init__(self):
        self.detector = RetinaFace

    def detect_faces(self, image):
        faces = self.detector.detect_faces(image)
        if isinstance(faces, tuple) or faces is None:
            return []
        return [{'box': [int(faces[key]['facial_area'][0]), int(faces[key]['facial_area'][1]),
                         int(faces[key]['facial_area'][2] - faces[key]['facial_area'][0]),
                         int(faces[key]['facial_area'][3] - faces[key]['facial_area'][1])],
                 'confidence': faces[key]['score']} for key in faces]

    def extract_face(self, image, box):
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]
        if face.size == 0:
            return None
        return face

class FaceRecognizer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.face_size = (224, 224)
        self.load_model()

    def preprocess_face(self, face_image):
        if face_image is None or face_image.size == 0:
            return None
        face = cv2.resize(face_image, self.face_size)
        face = face.astype('float32') / 255.0
        face = (face - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return face

    def load_model(self):
        self.model = tf.keras.models.load_model('resnet50_face_model.h5')
        with open('resnet50_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

    def recognize_face(self, face_image):
        processed_face = self.preprocess_face(face_image)
        if processed_face is None:
            return "Unknown", 0.0
        prediction = self.model.predict(np.expand_dims(processed_face, axis=0))
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        name = self.label_encoder.inverse_transform([class_idx])[0]
        return name, confidence

detector = FaceDetector()
recognizer = FaceRecognizer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nFailed to open video.\r\n'
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        faces = detector.detect_faces(frame)
        
        for face in faces[:5]:  # Limit to 5 people
            if face['confidence'] > 0.95:
                x, y, w, h = face['box']
                face_image = detector.extract_face(frame, face['box'])
                name, confidence = recognizer.recognize_face(face_image)
                label = f"{name} ({confidence:.2f})"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index_recognition.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index_recognition.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index_recognition.html', error="No file selected.")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        global current_video_path
        current_video_path = file_path
        return render_template('result_recognition.html')
    
    return render_template('index_recognition.html', error="Invalid file type. Use MP4, AVI, or MOV.")

@app.route('/video_feed')
def video_feed():
    global current_video_path
    if not current_video_path or not os.path.exists(current_video_path):
        return "No video available", 404
    return Response(generate_video_frames(current_video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

current_video_path = None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)