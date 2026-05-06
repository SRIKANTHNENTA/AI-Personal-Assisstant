"""
Emotion Analyzer
Multi-modal emotion detection from text, voice, and facial expressions
"""

import cv2
import numpy as np
from typing import Dict, Optional
from deepface import DeepFace
from django.conf import settings


class EmotionAnalyzer:
    """Analyzes emotions from multiple sources"""
    
    def __init__(self):
        self.emotion_mapping = {
            'angry': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprised',
            'neutral': 'neutral'
        }
    
    def analyze_facial_emotion(self, image_path: str) -> Dict[str, any]:
        """
        Analyze emotion from facial image using DeepFace
        """
        try:
            # Analyze face
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                enforce_detection=False
            )
            
            if isinstance(result, list):
                result = result[0]
            
            # Get dominant emotion
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            
            # Map to our emotion labels
            mapped_emotion = self.emotion_mapping.get(dominant_emotion, 'neutral')
            
            # Calculate confidence (percentage of dominant emotion)
            confidence = emotions.get(dominant_emotion, 0) / 100.0
            
            return {
                'success': True,
                'primary_emotion': mapped_emotion,
                'primary_confidence': confidence,
                'all_emotions': emotions,
                'facial_area': result.get('region', {}),
                'method': 'deepface'
            }
            
        except Exception as e:
            print(f"Facial emotion analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'primary_emotion': 'neutral',
                'primary_confidence': 0.0
            }
    
    def analyze_text_emotion(self, text: str) -> Dict[str, any]:
        """
        Analyze emotion from text using NLP
        """
        from .nlp_engine import NLPEngine
        
        nlp = NLPEngine()
        result = nlp.detect_emotion_from_text(text)
        
        return {
            'success': True,
            'emotion': result['emotion'],
            'intensity': result['intensity'],
            'confidence': result['confidence'],
            'sentiment_scores': result['sentiment_scores'],
            'method': 'text_analysis'
        }
    
    def analyze_voice_emotion(self, audio_file_path: str) -> Dict[str, any]:
        """
        Analyze emotion from voice tone
        Placeholder - implement with librosa and ML model
        """
        try:
            # This is a placeholder implementation
            # In production, use librosa to extract features and ML model to predict emotion
            
            # For now, return neutral with low confidence
            return {
                'success': True,
                'emotion': 'neutral',
                'confidence': 0.3,
                'voice_features': {
                    'pitch': 0.0,
                    'energy': 0.0,
                    'speech_rate': 0.0
                },
                'method': 'voice_analysis',
                'note': 'Placeholder - implement with librosa and ML model'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'emotion': 'neutral',
                'confidence': 0.0
            }
    
    def combine_emotion_sources(self, 
                                text_emotion: Dict = None,
                                voice_emotion: Dict = None,
                                facial_emotion: Dict = None) -> Dict[str, any]:
        """
        Combine emotions from multiple sources with weighted average
        """
        emotions = []
        weights = []
        
        # Text emotion (weight: 0.3)
        if text_emotion and text_emotion.get('success'):
            emotions.append(text_emotion['emotion'])
            weights.append(0.3 * text_emotion.get('confidence', 0.5))
        
        # Voice emotion (weight: 0.3)
        if voice_emotion and voice_emotion.get('success'):
            emotions.append(voice_emotion['emotion'])
            weights.append(0.3 * voice_emotion.get('confidence', 0.5))
        
        # Facial emotion (weight: 0.4 - most reliable)
        if facial_emotion and facial_emotion.get('success'):
            emotions.append(facial_emotion['primary_emotion'])
            weights.append(0.4 * facial_emotion.get('primary_confidence', 0.5))
        
        if not emotions:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'sources_used': []
            }
        
        # Find most confident emotion
        max_weight_idx = weights.index(max(weights))
        final_emotion = emotions[max_weight_idx]
        final_confidence = max(weights)
        
        return {
            'emotion': final_emotion,
            'confidence': final_confidence,
            'sources_used': [
                'text' if text_emotion else None,
                'voice' if voice_emotion else None,
                'facial' if facial_emotion else None
            ],
            'individual_results': {
                'text': text_emotion,
                'voice': voice_emotion,
                'facial': facial_emotion
            }
        }


class CameraEmotionDetector:
    """Real-time emotion detection from camera feed"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_analyzer = EmotionAnalyzer()
    
    def detect_face_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face from video frame
        Returns cropped face image or None
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get first face
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                return face_img
            
            return None
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None
    
    def analyze_frame_emotion(self, frame: np.ndarray, temp_file_path: str) -> Dict[str, any]:
        """
        Analyze emotion from a video frame
        """
        try:
            # Detect face
            face = self.detect_face_from_frame(frame)
            
            if face is None:
                return {
                    'success': False,
                    'error': 'No face detected',
                    'emotion': 'neutral',
                    'confidence': 0.0
                }
            
            # Save face to temp file
            cv2.imwrite(temp_file_path, face)
            
            # Analyze emotion
            result = self.emotion_analyzer.analyze_facial_emotion(temp_file_path)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'emotion': 'neutral',
                'confidence': 0.0
            }
