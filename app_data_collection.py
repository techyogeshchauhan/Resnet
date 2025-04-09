import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, Response
from datetime import datetime
import shutil

app = Flask(__name__)

# Configuration
DATA_FOLDER = 'dataset'
TEMP_VIDEO_FOLDER = 'temp_videos'
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(TEMP_VIDEO_FOLDER, exist_ok=True)

# Global variables
recording = False
current_name = None
current_dob = None
cap = None

# Generate video frames for live preview
def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Record video and save frames
def record_and_save_frames(name, dob):
    global recording, cap
    recording = True
    
    output_folder = os.path.join(DATA_FOLDER, f"{name}_{dob}")
    temp_video_path = os.path.join(TEMP_VIDEO_FOLDER, f"{name}_{dob}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    os.makedirs(output_folder, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640, 480))
    
    print(f"Recording for {name} (DOB: {dob})...")
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < 10:  # Record for 10 seconds
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)
    
    out.release()
    recording = False
    print("Recording stopped.")
    
    # Extract frames from video
    vidcap = cv2.VideoCapture(temp_video_path)
    frame_count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, image)
        frame_count += 1
    
    vidcap.release()
    os.remove(temp_video_path)  # Clean up temp video
    print(f"Saved {frame_count} frames to {output_folder}")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_recording():
    global current_name, current_dob
    current_name = request.form['name']
    current_dob = request.form['dob']
    
    if not current_name or not current_dob:
        return render_template('index.html', error="Please provide both name and date of birth.")
    
    return render_template('record.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record', methods=['POST'])
def record():
    global recording, current_name, current_dob
    if current_name and current_dob and not recording:
        record_and_save_frames(current_name, current_dob)
        return redirect(url_for('index'))
    return render_template('record.html', error="Recording failed or already in progress.")

@app.route('/stop')
def stop():
    global cap
    if cap is not None:
        cap.release()
    return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        if 'cap' in globals() and cap is not None:
            cap.release()