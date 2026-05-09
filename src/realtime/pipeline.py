import cv2
import joblib
import json
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


scaler = joblib.load(r"C:\Users\HP\Drowsiness-Project\feature_extraction\extracted_features\scaler.pkl")
svm_model = joblib.load(r"C:\Users\HP\Drowsiness-Project\model\svm_model.pkl")

label_map = json.load(open(r"C:\Users\HP\Drowsiness-Project\feature_extraction\extracted_features\label_map.json"))
label_map = {v: k for k, v in label_map.items()}


base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(base.input, GlobalAveragePooling2D()(base.output))


def extract_features(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32")
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)
    return feature_extractor.predict(frame, verbose=0)[0]


def predict(feature_vector):
    X = scaler.transform([feature_vector])
    return svm_model.predict(X)[0]


class DrowsinessPipeline:

    def process(self, frame):

        features = extract_features(frame)
        prediction = predict(features)
        label = label_map[prediction]

        return prediction, label