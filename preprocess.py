import cv2
import numpy as np

def preprocess_image(file):
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img