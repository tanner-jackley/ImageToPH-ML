import cv2
import numpy as np

def load_image(path: str):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return image

def convert_to_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

def extract_hsv_features(image):
    h, s, v = cv2.split(image)

    mean_h = np.mean(h)
    mean_s = np.mean(s)
    mean_v = np.mean(v)

    return np.array([mean_h, mean_s, mean_v])

def normalize_hsv(features):
    h, s, v = features
    return np.array([
        h / 179.0,
        s / 255.0,
        v / 255.0
    ])

def preprocess_image(path: str):
    image = load_image(path)
    hsv_image = convert_to_hsv(image)
    features = extract_hsv_features(hsv_image)
    normalized_hsv = normalize_hsv(features)

    return normalized_hsv