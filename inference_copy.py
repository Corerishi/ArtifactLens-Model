import tensorflow as tf
import tf_keras
import numpy as np
import cv2
from mtcnn import MTCNN
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "models/artifact_lens_final.h5"
model = tf_keras.models.load_model(MODEL_PATH)


class FaceProcessor:
    def __init__(self):
        self.detector = MTCNN()

    def align_and_crop(self, img, output_size=(224, 224)):
        if img is None: return None
        
        h, w = img.shape[:2]
        if h < 80 or w < 80:
            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.detector.detect_faces(img_rgb)
        except Exception as e:
            return None
        
        if not results:
            return None

        face = results[0]
        keypoints = face['keypoints']
        le, re = keypoints['left_eye'], keypoints['right_eye']

        dY = re[1] - le[1]
        dX = re[0] - le[0]
        angle = np.degrees(np.arctan2(dY, dX))
        center = (float((le[0] + re[0]) / 2), float((le[1] + re[1]) / 2))
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

        x, y, bw, bh = face['box']
        margin = 30
        y1, y2 = max(0, y - margin), min(h, y + bh + margin)
        x1, x2 = max(0, x - margin), min(w, x + bw + margin)
        
        crop = rotated_img[y1:y2, x1:x2]
        
        if crop.size == 0: return None
        return cv2.resize(crop, output_size)


processor = FaceProcessor()


def analyze_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_img = processor.align_and_crop(img)

    if face_img is None:
        return {"error": "No face detected"}

    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    prediction = model.predict(face_img)
    print(f"Raw prediction value: {prediction[0][0]}")
    if prediction[0][0] > 0.35:
        label = "Real"
        confidence = round(float(prediction[0][0]), 2)
    else:
        label = "Fake"
        confidence = round(float(1 - prediction[0][0]), 2)
    

    return {"label": label, "confidence": confidence}