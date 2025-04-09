import cv2
import numpy as np
from retinaface import RetinaFace
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FaceDetector:
    def __init__(self):
        self.detector = RetinaFace

    def detect_faces(self, image):
        try:
            faces = self.detector.detect_faces(image)
            if isinstance(faces, tuple) or faces is None:
                return []
            return [{'box': [int(faces[key]['facial_area'][0]), int(faces[key]['facial_area'][1]),
                             int(faces[key]['facial_area'][2] - faces[key]['facial_area'][0]),
                             int(faces[key]['facial_area'][3] - faces[key]['facial_area'][1])],
                     'confidence': faces[key]['score']} for key in faces]
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

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
        self.label_encoder = LabelEncoder()
        self.face_size = (224, 224)

    def preprocess_face(self, face_image):
        if face_image is None or face_image.size == 0:
            return None
        face = cv2.resize(face_image, self.face_size)
        face = face.astype('float32') / 255.0
        face = (face - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return face

    def create_model(self, num_classes):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def load_dataset(self, folder_path):
        X, y = [], []
        detector = FaceDetector()
        all_folders = os.listdir(folder_path)
        
        for user_folder in tqdm(all_folders, desc="Loading dataset"):
            user_path = os.path.join(folder_path, user_folder)
            if not os.path.isdir(user_path):
                continue
            
            for image_name in os.listdir(user_path):
                image_path = os.path.join(user_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                faces = detector.detect_faces(image)
                if faces:
                    face = detector.extract_face(image, faces[0]['box'])
                    processed_face = self.preprocess_face(face)
                    if processed_face is not None:
                        X.append(processed_face)
                        y.append(user_folder.split('_')[0])  # Use name as label
        
        if len(X) == 0:
            raise ValueError("No valid faces found in dataset")
        return np.array(X), np.array(y)

    def train_model(self, data_path):
        X, y = self.load_dataset(data_path)
        
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes)
        
        # Split into train and val (80-20 split)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)
        
        self.model = self.create_model(num_classes)
        print("Training model...")
        self.model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      epochs=10,
                      batch_size=32)
        
        self.model.save('resnet50_face_model.h5')
        with open('resnet50_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("Model and encoder saved.")

def main():
    data_path = "D:\\face\\dataset"
    recognizer = FaceRecognizer()
    recognizer.train_model(data_path)

if __name__ == "__main__":
    main()