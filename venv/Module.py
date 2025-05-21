import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class ImageProcessor:

    def __init__(self):
        self.model = ResNet50(weights='imagenet')
        self.target_size = (244, 244)

        self.scene_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
        self.scene_model_input_size = (244, 244)

        self.scene_detector = cv2.createBackgroundSubtractorMOG2()
        self.object_detector = cv2.createBackgroundSubtractorKNN()
        self.object_detector.setDetectShadows(True)

        self.object_detection_model = hub.load("https://tf.hub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
        self.object_detection_input_size = (320, 320)
        self.coco_labels = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
            'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.sentiment_model = tf.keras.models.load_model('path_to_your_sentiment_model.h5')
        self.sentiment_labels = ['positive', 'negative', 'neutral']

    def _preprocess_image_for_model(self, img, target_size):
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_normalized = img_rgb.astype(np.float32) / 255.0

        img_tensor = np.expand_dims(img_normalized, axis=0)

        return img_tensor

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
    
    def _calculate_colorfulness(self, img) -> float:
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
    
    def _get_scene_tags(self, img) -> list:
        preprocessed_img = self._preprocess_image_for_model(img, self.scene_model_input_size)

        if hasattr(self, 'scene_model'):
            predictions = self.scene_model.predict(preprocessed_img)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            scene_tags = [pred[1] for pred in decoded_predictions]
            return scene_tags
        else:
            print("Scene model not loaded. Using default tags.")
        if np.mean(cv2.CvtColor(img, cv2.COLOR_BGR2GRAY)) > 150:
            return ["bright_scene"]
        else:
            return ["dark_scene"]
        
    def _load_coco_labels(self) -> dict:
        labels_path = tf.keras.utils.get_file('coco_labels.txt', 'https://storage.googleapis.com/tensorflow/models/ssd_mobilenet_v2/coco_labels.txt')
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
        return {i: label for i, label in enumerate(labels)}
    
    def _get_object_tags(self, img) -> list:
        if hasattr(self, 'object_detection_model'):
            preprocessed_img = self._preprocess_image_for_model(img, self.object_detection_input_size)
            detections = self.object_detection_model(preprocessed_img)

            num_detections = int(detections.pop('num_detections'))
            detection_classses = detections['detection_classes'][0].numpy().astype(np.int32)[:num_detections]
            detection_score = detections['detection_scores'][0].numpy()[:num_detections]

            object_tags = set() 
            confidence_threshold = 0.5

            for i in range(num_detections):
                if detection_score[i] > confidence_threshold:
                    class_id = detection_classses[i]
                    if 0 <= class_id < len(self.coco_labels):
                        label = self.coco_labels[class_id]
                        object_tags.add(label)
                    else:
                        print(f"Class ID {class_id} is out of range for COCO labels.")

            return object_tags if object_tags else ["no_object_detected_by_model"]
        else:
            print("Object detection model not loaded. Using default tags.")

        fg_mask  = self.object_detector.apply(img)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_tags = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                object_tags.append("object_detected")
        return object_tags if object_tags else ["no_object_detected"]
    
    def _get_image_sentiment(self, img) -> str:
        sentiments = ["positive", "negative", "neutral"]
        return np.random.choice(sentiments)

class Song:
    def __init__(self):
        self.title = None
        self.artist = None
        self.genre = None
        self.lyrics = None
        self.audio_features = None
        self.mood = None
        self.energy_level = None
        self.themes = []

    def set_title(self, title: str):
        self.title = title
    
    def set_artist(self, artist: str):
        self.artist = artist

    def set_genre(self, genre: str):
        self.genre = genre

    def set_lyrics(self, lyrics: str):
        self.lyrics = lyrics

    def set_audio_features(self, audio_features: dict):
        self.audio_features = audio_features

    def set_mood(self, mood: str):
        self.mood = mood

    def set_energy_level(self, energy_level: str):
        self.energy_level = energy_level
    
    def add_theme(self, theme: str):
        if theme not in self.themes:
            self.themes.append(theme)

    def __repr__(self):
        return (f"Song(title={self.title}, artist={self.artist}, genre={self.genre}, "
                f"mood={self.mood}, energy_level={self.energy_level}, themes={self.themes})")
    
def main():
    processor = ImageProcessor()
    



if __name__ ==  "__main__":
    main()