Below is a detailed `README.md` file that provides step-by-step instructions to run the three parts of your project: **Data Collection**, **Training**, and **Detection/Recognition**. This assumes you’re working in a Windows environment (since your paths use `D:\face`) and includes prerequisites, setup, and troubleshooting tips.

---

# Face Recognition Project README

This project consists of three parts:
1. **Data Collection**: Collects face data from different angles using a webcam and saves it as frames.
2. **Training**: Trains a ResNet50 model with RetinaFace for face recognition using the collected data.
3. **Detection and Recognition**: Detects and recognizes up to 5 people in a video simultaneously.

## Prerequisites
- **Operating System**: Windows (tested on paths like `D:\face`)
- **Hardware**: Webcam (for data collection), decent CPU/GPU (for training and recognition)
- **Software**:
  - Python 3.9 (recommended for compatibility)
  - Miniconda or Anaconda (for environment management)
  - Git (optional, for cloning repositories)

## Directory Structure
Create the following structure in `D:\face`:
```
D:\face\
├── dataset\              # Where collected frames are stored
├── uploads\              # For uploaded videos in recognition
├── static\output\        # For output files in recognition
├── templates\            # HTML templates
│   ├── index.html
│   ├── record.html
│   ├── index_recognition.html
│   └── result_recognition.html
├── app_data_collection.py
├── train_resnet50.py
├── app_recognition.py
├── resnet50_face_model.h5  # Generated after training
└── resnet50_label_encoder.pkl  # Generated after training
```

## Step-by-Step Instructions

### Part 1: Data Collection

#### Setup
1. **Create a Conda Environment**:
   ```bash
   conda create -n face_recognition python=3.9
   conda activate face_recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install flask opencv-python
   ```

3. **Save the Code**:
   - Save `app_data_collection.py` from the provided code in `D:\face`.
   - Create a `templates` folder in `D:\face` and save `index.html` and `record.html` there.

#### Running
1. **Start the Flask App**:
   ```bash
   cd D:\face
   python app_data_collection.py
   ```

2. **Access the App**:
   - Open your browser and go to `http://127.0.0.1:5000`.
   - Enter a name (e.g., "John") and date of birth (e.g., "1990-01-01").
   - Click "Start Data Collection".

3. **Record Video**:
   - You’ll see a live webcam feed. Rotate your head (up, down, left, right) for 10 seconds.
   - Click "Save Data" to stop recording and save frames.

4. **Output**:
   - Frames are saved in `D:\face\dataset\{name}_{dob}\frame_xxxx.jpg` (e.g., `D:\face\dataset\John_1990-01-01\frame_0001.jpg`).

#### Troubleshooting
- **Webcam Not Working**: Ensure your webcam is connected and not in use by another app. Check the console for "Could not open webcam" errors.
- **Frames Not Saving**: Verify write permissions in `D:\face\dataset`.

---

### Part 2: Training

#### Setup
1. **Activate Environment**:
   ```bash
   conda activate face_recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install tensorflow==2.10.0 retinaface numpy==1.23.5 scikit-learn tqdm opencv-python
   ```

3. **Prepare Dataset**:
   - Ensure you’ve collected data for multiple people in `D:\face\dataset` (e.g., `John_1990-01-01`, `Jane_1992-02-02`).

4. **Save the Code**:
   - Save `train_resnet50.py` in `D:\face`.

#### Running
1. **Train the Model**:
   ```bash
   cd D:\face
   python train_resnet50.py
   ```

2. **Process**:
   - The script loads frames, detects faces with RetinaFace, preprocesses them, and trains a ResNet50 model.
   - Training runs for 10 epochs (adjustable in the script).

3. **Output**:
   - Saves `resnet50_face_model.h5` and `resnet50_label_encoder.pkl` in `D:\face`.

#### Troubleshooting
- **No Faces Found**: Ensure dataset folders contain valid images with detectable faces.
- **Memory Error**: Reduce `batch_size` in `train_resnet50.py` (e.g., from 32 to 16).
- **Dependency Conflict**: If errors occur, recreate the environment with exact versions specified above.

---

### Part 3: Detection and Recognition

#### Setup
1. **Activate Environment**:
   ```bash
   conda activate face_recognition
   ```

2. **Install Dependencies**:
   - If not already installed from previous steps:
     ```bash
     pip install flask tensorflow==2.10.0 retinaface numpy==1.23.5 opencv-python scikit-learn
     ```

3. **Prepare Model**:
   - Ensure `resnet50_face_model.h5` and `resnet50_label_encoder.pkl` are in `D:\face` from the training step.

4. **Save the Code**:
   - Save `app_recognition.py` in `D:\face`.
   - Create `templates` folder (if not already) and save `index_recognition.html` and `result_recognition.html`.

#### Running
1. **Start the Flask App**:
   ```bash
   cd D:\face
   python app_recognition.py
   ```

2. **Access the App**:
   - Open your browser and go to `http://127.0.0.1:5001`.
   - Upload a video file (MP4, AVI, or MOV) containing 1-5 people.

3. **View Results**:
   - The app streams the video with real-time face detection and recognition (up to 5 people).

#### Troubleshooting
- **Model Not Found**: Ensure `resnet50_face_model.h5` and `resnet50_label_encoder.pkl` are in `D:\face`.
- **Video Not Playing**: Check file format and ensure it’s not corrupted.
- **Slow Performance**: Reduce video resolution in `app_recognition.py` (e.g., change `(640, 480)` to `(320, 240)`).

---

## General Tips
- **Environment**: Always activate `face_recognition` before running any script:
  ```bash
  conda activate face_recognition
  ```
- **Ports**: Data Collection uses port 5000, Recognition uses 5001. Ensure they’re free.
- **Dataset Quality**: For best results, collect clear, well-lit videos with distinct head rotations.

## Dependencies Recap
- `flask`: Web framework
- `opencv-python`: Image/video processing
- `tensorflow==2.10.0`: Deep learning
- `retinaface`: Face detection
- `numpy==1.23.5`: Numerical operations
- `scikit-learn`: Data splitting and encoding
- `tqdm`: Progress bars

## Hinglish Note
Yeh README har step ko aasani se samjhayega! Pehle data collect karo, phir model train karo, aur akhir mein video mein 1-5 logon ko recognize karo. Sab step follow karo, agar koi error aaye toh mujhe pura message bhejo, main fix kar dunga!

---

Save this as `README.md` in `D:\face` and follow the instructions. Let me know if you encounter any issues!
