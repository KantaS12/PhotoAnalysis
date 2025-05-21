import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class ImageProcessor:

    def __init__(self):
        self.model = ResNet50(weights='imagenet')
        self.target_size = (244, 244)
        self.scence_detector = cv2.createBackgroundSubtractorMOG2()
        self.object_detector = cv2.createBackgroundSubtractorKNN()
        self.object_detector.setDetectShadows(True)

    def extract_features(self, image_path: str, showFeatures = False) -> dict:

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        features = {}

        #Dominant Colors (using KMMeans)
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)

        #Apply KMeans

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10 , cv2.KMEANS_RANDOM_CENTERS)
        #Percentage of each color
        _, counts = np.unique(labels, return_counts=True)
        dominant_colors = [(centers[i].astype(int).tolist(), count / sum(counts)) for i, count in enumerate(counts)]
        features['dominant_colors'] = dominant_colors #[(RGB, percentage),...]

        #Average Brightness
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features['average_brightness'] = np.mean(gray_img)

        #Colorfulness
        features['colorfulness'] = self._calculate_colorfulness(img)

        #Scene Tags
        features['scene_tags'] = self._get_scene_tags(img)

        #Object Tags
        features['object_tags'] = self._get_object_tags(img)

        #Image Sentiment
        features['image_sentiment'] = self._get_image_sentiment(img)

        return features
    
    def _calculate_colorfulness(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a_std = np.std(a)
        b_std = np.std(b)
        colorfulness = np.sqrt((a_std ** 2) + (b_std ** 2)) + (0.3 * np.sqrt((a_mean ** 2) + (b_mean ** 2)))

        return colorfulness
    

class Song:
    pass