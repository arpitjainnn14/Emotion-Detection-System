from deepface import DeepFace
import numpy as np
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.previous_emotions = []
        self.smooth_window = 2  # Reduced window for faster response
        self.emotion_weights = {
            'sad': 1.2,      # Increase sensitivity to sadness
            'disgust': 1.2,   # Increase sensitivity to disgust
            'angry': 1.1,     # Slightly increase sensitivity to anger
            'fear': 1.1,      # Slightly increase sensitivity to fear
            'happy': 0.9,     # Slightly decrease sensitivity to happiness
            'neutral': 0.8,   # Decrease sensitivity to neutral
            'surprise': 1.0   # Keep surprise as is
        }
        
    def enhance_contrast(self, img):
        """Enhance image contrast to better detect subtle expressions."""
        if len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl,a,b))
            
            # Convert back to BGR
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return img

    def analyze_emotion(self, face_img):
        """
        Analyze emotion in the face image with enhanced sensitivity.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            tuple: (dominant_emotion, confidence)
        """
        try:
            # Basic validation
            if face_img is None or face_img.size == 0:
                return 'neutral', 0.0

            # Ensure minimum size
            min_size = 96  # Increased minimum size for better detail
            if face_img.shape[0] < min_size or face_img.shape[1] < min_size:
                face_img = cv2.resize(face_img, (min_size, min_size))

            # Enhance contrast
            face_img = self.enhance_contrast(face_img)

            # Try multiple analysis attempts with different settings
            results = []
            try:
                # First attempt with default settings
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                results.append(result[0]['emotion'])
            except Exception as e:
                logger.warning(f"First attempt failed: {str(e)}")

            try:
                # Second attempt with opencv backend
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                results.append(result[0]['emotion'])
            except Exception as e:
                logger.warning(f"Second attempt failed: {str(e)}")

            if not results:
                return 'neutral', 0.0

            # Average the results from different attempts
            avg_emotions = {}
            for emotion in self.emotions:
                values = [r.get(emotion, 0) for r in results]
                avg_emotions[emotion] = sum(values) / len(values)

            # Apply emotion weights
            weighted_emotions = {
                emo: conf * self.emotion_weights.get(emo, 1.0)
                for emo, conf in avg_emotions.items()
            }
            
            # Get dominant emotion
            dominant_emotion = max(weighted_emotions.items(), key=lambda x: x[1])
            
            # Apply temporal smoothing with reduced window
            self.previous_emotions.append(dominant_emotion[0])
            if len(self.previous_emotions) > self.smooth_window:
                self.previous_emotions.pop(0)
            
            # Get the most frequent emotion in the window
            from collections import Counter
            if self.previous_emotions:
                smoothed_emotion = Counter(self.previous_emotions).most_common(1)[0][0]
                # Use the original (non-weighted) confidence
                smoothed_confidence = avg_emotions[smoothed_emotion] / 100.0
            else:
                smoothed_emotion = dominant_emotion[0]
                smoothed_confidence = avg_emotions[smoothed_emotion] / 100.0

            # Additional check for subtle emotions
            if smoothed_emotion == 'neutral' and smoothed_confidence < 0.6:
                # Look for the next highest emotion
                other_emotions = {k: v for k, v in avg_emotions.items() if k != 'neutral'}
                if other_emotions:
                    next_emotion = max(other_emotions.items(), key=lambda x: x[1])
                    if next_emotion[1] > 20:  # If any other emotion has >20% confidence
                        return next_emotion[0], next_emotion[1] / 100.0
            
            return smoothed_emotion, smoothed_confidence
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return 'neutral', 0.1
    
    def get_emotion_color(self, emotion):
        """
        Get color for emotion visualization.
        
        Args:
            emotion: Emotion name
            
        Returns:
            tuple: BGR color values
        """
        color_map = {
            'happy': (0, 255, 255),     # Yellow
            'sad': (255, 128, 0),       # Orange
            'angry': (0, 0, 255),       # Red
            'surprise': (255, 255, 0),   # Cyan
            'fear': (255, 0, 255),      # Magenta
            'disgust': (0, 255, 0),     # Green
            'neutral': (255, 255, 255),  # White
            'unknown': (128, 128, 128)   # Gray
        }
        return color_map.get(emotion, (128, 128, 128))
    
    def get_emotion_emoji(self, emotion):
        """
        Get emoji for emotion visualization.
        
        Args:
            emotion: Emotion name
            
        Returns:
            str: Emoji character
        """
        emoji_map = {
            'happy': '😄',
            'sad': '😢',
            'angry': '😡',
            'surprise': '😮',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐',
            'unknown': '❓'
        }
        return emoji_map.get(emotion, '❓')
    
    def get_emotion_description(self, emotion, confidence):
        """
        Get a description of the emotion.
        
        Args:
            emotion: Emotion name
            confidence: Confidence score
            
        Returns:
            str: Description of the emotion
        """
        descriptions = {
            'happy': f"Happy ({confidence:.0%} confidence)",
            'sad': f"Sad ({confidence:.0%} confidence)",
            'angry': f"Angry ({confidence:.0%} confidence)",
            'surprise': f"Surprised ({confidence:.0%} confidence)",
            'fear': f"Afraid ({confidence:.0%} confidence)",
            'disgust': f"Disgusted ({confidence:.0%} confidence)",
            'neutral': f"Neutral ({confidence:.0%} confidence)",
            'unknown': "Unable to determine emotion"
        }
        return descriptions.get(emotion, "Unable to determine emotion") 