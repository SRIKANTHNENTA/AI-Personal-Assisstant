"""
Advanced Multimodal Emotion Detector
Real-time emotion detection from text, voice, and facial expressions using deep learning.
Implements CNN for facial analysis, BERT for text, and LSTM for voice emotion.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import warnings

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*The name tf\.losses\.sparse_softmax_cross_entropy is deprecated.*",
)


class MultimodalEmotionDetector:
    """
    Advanced multimodal emotion detection system.
    Integrates text, voice, and facial emotion analysis using deep learning models.
    """
    
    EMOTION_CLASSES = [
        'angry', 'disgusted', 'fearful', 'happy', 
        'neutral', 'sad', 'surprised', 'excited', 
        'anxious', 'calm', 'stressed', 'tired'
    ]
    
    VALENCE_AROUSAL_MAP = {
        'happy': (0.8, 0.6),
        'excited': (0.7, 0.9),
        'calm': (0.5, 0.2),
        'neutral': (0.0, 0.3),
        'sad': (-0.7, 0.2),
        'angry': (-0.6, 0.8),
        'fearful': (-0.5, 0.7),
        'anxious': (-0.4, 0.7),
        'stressed': (-0.5, 0.8),
        'surprised': (0.3, 0.8),
        'disgusted': (-0.6, 0.5),
        'tired': (-0.2, 0.1)
    }
    
    def __init__(self):
        # Lazy initialization attributes
        self._text_pipeline = None
        self._deepface = None
        self.voice_model = None
        
        self.models_loaded = {
            'text': False,
            'voice': False,
            'facial': False
        }
        
        # Model paths
        self.model_dir = Path(__file__).parent / 'emotion_models'
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize voice model (fast)
        self._init_voice_model()
    
    def preload_all_models(self):
        """Pre-load all AI models to reduce first-response latency"""
        print("🚀 Pre-loading Multimodal AI models...")
        _ = self.text_model
        _ = self.deepface_mod
        print("✅ All Multimodal models pre-loaded.")
    
    @property
    def text_model(self):
        """Lazy loader for BERT-based text emotion model"""
        if self._text_pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                print("⏳ Loading Text emotion model (RoBERTa)...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._text_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=tokenizer,
                    return_all_scores=True,
                    device=-1,
                )
                self.models_loaded['text'] = True
                print("✅ Text emotion model loaded.")
            except Exception as e:
                print(f"⚠️ Could not load text model: {e}")
                self.models_loaded['text'] = False
        return self._text_pipeline

    @property
    def deepface_mod(self):
        """Lazy loader for facial emotion model"""
        if self._deepface is None:
            try:
                from deepface import DeepFace
                print("⏳ Loading Facial emotion model (DeepFace)...")
                self._deepface = DeepFace
                self.models_loaded['facial'] = True
                print("✅ Facial emotion model loaded.")
            except ImportError:
                print("⚠️ DeepFace not installed.")
                self.models_loaded['facial'] = False
        return self._deepface

    def _init_voice_model(self):
        """Initialize voice emotion model"""
        try:
            import librosa
            self.librosa_available = True
            self.models_loaded['voice'] = True
            print("✅ Voice emotion processing available (librosa)")
        except ImportError:
            print("⚠️ librosa not installed. Voice emotion disabled.")
            print("Install with: pip install librosa")
            self.librosa_available = False
    
    def detect_text_emotion(self, text: str) -> Dict:
        """
        Detect emotion from text using BERT/RoBERTa.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with emotion, confidence, valence, arousal
        """
        if not text or not text.strip():
            return self._empty_result('text', 'Empty text input')
        
        try:
            if self.text_model:
                # Use transformer model
                raw_output = self.text_model(text[:512])  # Limit to 512 tokens

                # Hugging Face pipelines may return:
                # 1) list[dict] for a single label
                # 2) list[list[dict]] for return_all_scores=True
                # 3) dict in some custom wrappers
                if isinstance(raw_output, list):
                    if raw_output and isinstance(raw_output[0], list):
                        results = raw_output[0]
                    else:
                        results = raw_output
                elif isinstance(raw_output, dict):
                    results = [raw_output]
                else:
                    results = []
                
                # Map sentiment to emotions
                scores = {
                    r.get('label', '').lower(): float(r.get('score', 0.0))
                    for r in results
                    if isinstance(r, dict) and r.get('label')
                }

                if not scores:
                    return self._vader_emotion(text)
                
                # Convert sentiment to emotion
                if 'positive' in scores or 'pos' in scores:
                    pos_score = scores.get('positive', scores.get('pos', 0))
                    neg_score = scores.get('negative', scores.get('neg', 0))
                    neu_score = scores.get('neutral', scores.get('neu', 0))
                    
                    if pos_score > neg_score and pos_score > neu_score:
                        emotion = 'happy' if pos_score > 0.7 else 'calm'
                        confidence = pos_score
                    elif neg_score > pos_score and neg_score > neu_score:
                        emotion = 'sad' if neg_score > 0.7 else 'anxious'
                        confidence = neg_score
                    else:
                        emotion = 'neutral'
                        confidence = neu_score
                else:
                    # Direct emotion scores
                    best_emotion = max(scores, key=scores.get)
                    emotion = best_emotion
                    confidence = scores[best_emotion]
                
                valence, arousal = self.VALENCE_AROUSAL_MAP.get(emotion, (0.0, 0.5))
                
                return {
                    'success': True,
                    'emotion': emotion,
                    'confidence': confidence,
                    'valence': valence,
                    'arousal': arousal,
                    'all_scores': scores,
                    'model': 'roberta'
                }
            else:
                # Fallback to VADER
                return self._vader_emotion(text)
                
        except Exception as e:
            print(f"Text emotion error: {e}")
            return self._vader_emotion(text)
    
    def _vader_emotion(self, text: str) -> Dict:
        """Fallback VADER-based emotion detection"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            
            vader = SentimentIntensityAnalyzer()
            scores = vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.5:
                emotion = 'happy'
            elif compound >= 0.1:
                emotion = 'calm'
            elif compound >= -0.1:
                emotion = 'neutral'
            elif compound >= -0.5:
                emotion = 'sad'
            else:
                emotion = 'angry'
            
            valence, arousal = self.VALENCE_AROUSAL_MAP.get(emotion, (0.0, 0.5))
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': abs(compound),
                'valence': valence,
                'arousal': arousal,
                'all_scores': scores,
                'model': 'vader'
            }
        except Exception as e:
            return self._empty_result('text', str(e))
    
    def detect_voice_emotion(self, audio_path: str = None, 
                            audio_data: np.ndarray = None,
                            sample_rate: int = 22050) -> Dict:
        """
        Detect emotion from voice using acoustic features.
        
        Args:
            audio_path: Path to audio file
            audio_data: Raw audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            Dict with emotion, confidence, voice features
        """
        if not self.librosa_available:
            return self._empty_result('voice', 'librosa not available')
        
        try:
            import librosa
            import scipy.stats as stats
            
            # Load audio
            if audio_path:
                y, sr = librosa.load(audio_path, sr=sample_rate)
            elif audio_data is not None:
                y = audio_data
                sr = sample_rate
            else:
                return self._empty_result('voice', 'No audio input provided')
            
            # Extract features
            features = self._extract_voice_features(y, sr)
            
            # Rule-based emotion classification from features
            # (In production, use a trained LSTM/CNN model)
            emotion, confidence = self._classify_voice_emotion(features)
            valence, arousal = self.VALENCE_AROUSAL_MAP.get(emotion, (0.0, 0.5))
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'valence': valence,
                'arousal': arousal,
                'features': features,
                'model': 'acoustic_features'
            }
            
        except Exception as e:
            print(f"Voice emotion error: {e}")
            return self._empty_result('voice', str(e))
    
    def _extract_voice_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract acoustic features from audio"""
        import librosa
        
        features = {}
        
        # Pitch (F0) using librosa pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                features['pitch_mean'] = float(np.mean(f0_clean))
                features['pitch_std'] = float(np.std(f0_clean))
                features['pitch_range'] = float(np.ptp(f0_clean))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
        except:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        # Energy / RMS
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        features['energy_max'] = float(np.max(rms))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # MFCC (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        # Speech rate approximation
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        features['speech_rate'] = len(onset_frames) / duration if duration > 0 else 0
        
        return features
    
    def _classify_voice_emotion(self, features: Dict) -> Tuple[str, float]:
        """
        Classify emotion from voice features.
        Simple rule-based classifier (replace with ML model in production).
        """
        pitch_mean = features.get('pitch_mean', 0)
        pitch_std = features.get('pitch_std', 0)
        energy_mean = features.get('energy_mean', 0)
        speech_rate = features.get('speech_rate', 0)
        zcr_mean = features.get('zcr_mean', 0)
        
        # Normalize features (using typical ranges)
        normalized_pitch = min(pitch_mean / 300, 1.0) if pitch_mean > 0 else 0.5
        normalized_energy = min(energy_mean / 0.1, 1.0)
        normalized_speech_rate = min(speech_rate / 5, 1.0)
        
        # Simple rule-based classification
        if normalized_energy > 0.7 and normalized_pitch > 0.6:
            if pitch_std > 30:
                emotion = 'excited'
                confidence = 0.6
            else:
                emotion = 'angry'
                confidence = 0.5
        elif normalized_energy < 0.3:
            if normalized_pitch < 0.4:
                emotion = 'sad'
                confidence = 0.5
            else:
                emotion = 'tired'
                confidence = 0.4
        elif normalized_energy > 0.5 and normalized_speech_rate > 0.6:
            emotion = 'anxious'
            confidence = 0.5
        elif normalized_energy > 0.4 and normalized_pitch > 0.5:
            emotion = 'happy'
            confidence = 0.5
        else:
            emotion = 'neutral'
            confidence = 0.4
        
        return emotion, confidence
    
    def detect_facial_emotion(self, image_path: str = None,
                             image_data: np.ndarray = None) -> Dict:
        """
        Detect emotion from facial expression using DeepFace.
        
        Args:
            image_path: Path to image file
            image_data: Image as numpy array (BGR format)
            
        Returns:
            Dict with emotion, confidence, facial features
        """
        if not self.deepface_mod:
            return self._empty_result('facial', 'DeepFace not available')
        
        try:
            # Analyze face
            if image_path:
                result = self.deepface_mod.analyze(
                    img_path=image_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
            elif image_data is not None:
                result = self.deepface_mod.analyze(
                    img_path=image_data,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
            else:
                return self._empty_result('facial', 'No image input provided')
            
            if isinstance(result, list):
                result = result[0]
            
            # Extract emotion data
            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'neutral')
            
            # Map DeepFace emotions to our emotion set
            emotion_mapping = {
                'angry': 'angry',
                'disgust': 'disgusted',
                'fear': 'fearful',
                'happy': 'happy',
                'sad': 'sad',
                'surprise': 'surprised',
                'neutral': 'neutral'
            }
            
            emotion = emotion_mapping.get(dominant, 'neutral')
            confidence = emotions.get(dominant, 0) / 100.0
            valence, arousal = self.VALENCE_AROUSAL_MAP.get(emotion, (0.0, 0.5))
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'valence': valence,
                'arousal': arousal,
                'all_emotions': {
                    emotion_mapping.get(k, k): v / 100.0 
                    for k, v in emotions.items()
                },
                'facial_region': result.get('region', {}),
                'model': 'deepface'
            }
            
        except Exception as e:
            print(f"Facial emotion error: {e}")
            return self._empty_result('facial', str(e))
    
    def detect_multimodal_emotion(self,
                                  text: str = None,
                                  audio_path: str = None,
                                  audio_data: np.ndarray = None,
                                  image_path: str = None,
                                  image_data: np.ndarray = None) -> Dict:
        """
        Detect emotion from multiple modalities simultaneously.
        
        Args:
            text: Text input for analysis
            audio_path: Path to audio file
            audio_data: Raw audio data
            image_path: Path to facial image
            image_data: Facial image as numpy array
            
        Returns:
            Comprehensive multimodal emotion analysis
        """
        results = {
            'text': None,
            'voice': None,
            'facial': None
        }
        
        # Analyze each available modality
        if text:
            results['text'] = self.detect_text_emotion(text)
        
        if audio_path or audio_data is not None:
            results['voice'] = self.detect_voice_emotion(
                audio_path=audio_path,
                audio_data=audio_data
            )
        
        if image_path or image_data is not None:
            results['facial'] = self.detect_facial_emotion(
                image_path=image_path,
                image_data=image_data
            )
        
        # Check if any modality succeeded
        successful = {k: v for k, v in results.items() if v and v.get('success')}
        
        if not successful:
            return {
                'success': False,
                'error': 'No successful modality analysis',
                'modality_results': results
            }
        
        # Fuse results using weighted averaging
        fused = self._fuse_modalities(successful)
        
        return {
            'success': True,
            'primary_emotion': fused['emotion'],
            'confidence': fused['confidence'],
            'valence': fused['valence'],
            'arousal': fused['arousal'],
            'modality_results': results,
            'fusion_weights': fused.get('weights', {}),
            'modalities_used': list(successful.keys())
        }
    
    def _fuse_modalities(self, results: Dict) -> Dict:
        """
        Fuse emotion results from multiple modalities.
        Uses weighted averaging based on modality reliability.
        """
        # Default weights for each modality
        weights = {
            'text': 0.35,
            'voice': 0.25,
            'facial': 0.40
        }
        
        # Adjust weights based on confidence
        adjusted_weights = {}
        for modality, result in results.items():
            if result and result.get('success'):
                conf = result.get('confidence', 0.5)
                adjusted_weights[modality] = weights.get(modality, 0.33) * conf
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        # Weighted average of valence and arousal
        total_valence = 0.0
        total_arousal = 0.0
        total_confidence = 0.0
        
        emotion_votes = {}
        
        for modality, result in results.items():
            if result and result.get('success'):
                w = adjusted_weights.get(modality, 0)
                total_valence += w * result.get('valence', 0)
                total_arousal += w * result.get('arousal', 0.5)
                total_confidence += w * result.get('confidence', 0.5)
                
                emotion = result.get('emotion', 'neutral')
                emotion_votes[emotion] = emotion_votes.get(emotion, 0) + w
        
        # Get consensus emotion
        if emotion_votes:
            primary_emotion = max(emotion_votes, key=emotion_votes.get)
        else:
            primary_emotion = 'neutral'
        
        return {
            'emotion': primary_emotion,
            'confidence': total_confidence,
            'valence': total_valence,
            'arousal': total_arousal,
            'weights': adjusted_weights
        }
    
    def _empty_result(self, modality: str, error: str) -> Dict:
        """Return empty result with error"""
        return {
            'success': False,
            'emotion': 'neutral',
            'confidence': 0.0,
            'valence': 0.0,
            'arousal': 0.5,
            'error': error,
            'modality': modality
        }
    
    def get_model_status(self) -> Dict:
        """Get status of all loaded models"""
        return {
            'text_model': {
                'loaded': self.models_loaded['text'],
                'type': 'RoBERTa' if self.text_model else 'VADER (fallback)'
            },
            'voice_model': {
                'loaded': self.models_loaded['voice'],
                'type': 'Acoustic Features + Rules'
            },
            'facial_model': {
                'loaded': self.models_loaded['facial'],
                'type': 'DeepFace' if self._deepface else 'Not loaded/available'
            }
        }


class RealTimeEmotionTracker:
    """
    Real-time emotion tracking with temporal smoothing and trend analysis.
    Designed for continuous monitoring in WebSocket connections.
    """
    
    def __init__(self, detector: Optional[MultimodalEmotionDetector] = None, smooth_window: int = 5):
        self.detector = detector or MultimodalEmotionDetector()
        self.smooth_window = smooth_window
        self.emotion_buffer = []
        self.valence_buffer = []
        self.arousal_buffer = []
        self.timestamps = []
        
        self.max_buffer_size = 1000
    
    def update(self, text: str = None, audio_data: np.ndarray = None,
               image_data: np.ndarray = None) -> Dict:
        """
        Update emotion state with new data.
        
        Returns:
            Smoothed emotion state with trend information
        """
        # Detect current emotion
        result = self.detector.detect_multimodal_emotion(
            text=text,
            audio_data=audio_data,
            image_data=image_data
        )
        
        if not result.get('success'):
            return result
        
        # Add to buffers
        self.emotion_buffer.append(result['primary_emotion'])
        self.valence_buffer.append(result.get('valence', 0))
        self.arousal_buffer.append(result.get('arousal', 0.5))
        self.timestamps.append(datetime.now())
        
        # Trim buffers
        if len(self.emotion_buffer) > self.max_buffer_size:
            self.emotion_buffer = self.emotion_buffer[-self.max_buffer_size:]
            self.valence_buffer = self.valence_buffer[-self.max_buffer_size:]
            self.arousal_buffer = self.arousal_buffer[-self.max_buffer_size:]
            self.timestamps = self.timestamps[-self.max_buffer_size:]
        
        # Apply smoothing
        smoothed = self._apply_smoothing()
        
        # Calculate trend
        trend = self._calculate_trend()
        
        return {
            'success': True,
            'current': result,
            'smoothed': smoothed,
            'trend': trend,
            'buffer_size': len(self.emotion_buffer)
        }
    
    def _apply_smoothing(self) -> Dict:
        """Apply temporal smoothing to recent emotions"""
        if len(self.emotion_buffer) < 2:
            return {
                'emotion': self.emotion_buffer[-1] if self.emotion_buffer else 'neutral',
                'valence': self.valence_buffer[-1] if self.valence_buffer else 0,
                'arousal': self.arousal_buffer[-1] if self.arousal_buffer else 0.5
            }
        
        window = min(self.smooth_window, len(self.emotion_buffer))
        
        # Mode for emotion
        recent_emotions = self.emotion_buffer[-window:]
        emotion_counts = {}
        for e in recent_emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Mean for valence and arousal
        smoothed_valence = np.mean(self.valence_buffer[-window:])
        smoothed_arousal = np.mean(self.arousal_buffer[-window:])
        
        return {
            'emotion': smoothed_emotion,
            'valence': float(smoothed_valence),
            'arousal': float(smoothed_arousal),
            'consistency': emotion_counts[smoothed_emotion] / window
        }
    
    def _calculate_trend(self) -> Dict:
        """Calculate emotion trend over time"""
        if len(self.valence_buffer) < 5:
            return {'direction': 'stable', 'magnitude': 0.0}
        
        # Use last 10 points for trend
        recent_valence = np.array(self.valence_buffer[-10:])
        
        # Simple linear regression
        x = np.arange(len(recent_valence))
        slope = np.polyfit(x, recent_valence, 1)[0]
        
        if slope > 0.02:
            direction = 'improving'
        elif slope < -0.02:
            direction = 'declining'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'magnitude': abs(float(slope)),
            'slope': float(slope)
        }
    
    def get_emotion_summary(self, time_range_minutes: int = 60) -> Dict:
        """
        Get emotion summary for a time range.
        
        Args:
            time_range_minutes: Time range in minutes
            
        Returns:
            Summary statistics for the time period
        """
        if not self.timestamps:
            return {'success': False, 'error': 'No data available'}
        
        cutoff = datetime.now() - timedelta(minutes=time_range_minutes)
        
        # Filter data in range
        valid_indices = [
            i for i, t in enumerate(self.timestamps) 
            if t >= cutoff
        ]
        
        if not valid_indices:
            return {'success': False, 'error': 'No data in time range'}
        
        emotions = [self.emotion_buffer[i] for i in valid_indices]
        valences = [self.valence_buffer[i] for i in valid_indices]
        arousals = [self.arousal_buffer[i] for i in valid_indices]
        
        # Calculate statistics
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        total = len(emotions)
        emotion_distribution = {k: v / total for k, v in emotion_counts.items()}
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        return {
            'success': True,
            'time_range_minutes': time_range_minutes,
            'sample_count': total,
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_distribution,
            'average_valence': float(np.mean(valences)),
            'average_arousal': float(np.mean(arousals)),
            'valence_std': float(np.std(valences)),
            'arousal_std': float(np.std(arousals)),
            'emotional_stability': 1.0 - float(np.std(valences))
        }
    
    def reset(self):
        """Reset tracking buffers"""
        self.emotion_buffer = []
        self.valence_buffer = []
        self.arousal_buffer = []
        self.timestamps = []


class VoiceEmotionLSTM:
    """
    LSTM-based voice emotion recognition model.
    Trained on acoustic features extracted from speech.
    """
    
    def __init__(self, n_features: int = 40, n_classes: int = 8):
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.is_trained = False
        
        self.emotion_labels = [
            'angry', 'calm', 'fearful', 'happy',
            'neutral', 'sad', 'surprised', 'excited'
        ]
        
        self._build_model()
    
    def _build_model(self):
        """Build LSTM model for voice emotion"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                layers.LSTM(128, input_shape=(None, self.n_features),
                           return_sequences=True, dropout=0.3),
                layers.LSTM(64, return_sequences=False, dropout=0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.n_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Voice Emotion LSTM model built")
            
        except ImportError:
            print("⚠️ TensorFlow not available for Voice LSTM")
            self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> Dict:
        """Train the LSTM model"""
        if self.model is None:
            return {'success': False, 'error': 'Model not available'}
        
        try:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'success': True,
                'final_accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history.get('val_accuracy', [0])[-1]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict(self, features: np.ndarray) -> Dict:
        """Predict emotion from voice features"""
        if self.model is None or not self.is_trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            probs = self.model.predict(features, verbose=0)
            predicted_idx = np.argmax(probs[0])
            
            return {
                'success': True,
                'emotion': self.emotion_labels[predicted_idx],
                'confidence': float(probs[0][predicted_idx]),
                'all_probabilities': {
                    label: float(probs[0][i])
                    for i, label in enumerate(self.emotion_labels)
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class FacialEmotionCNN:
    """
    CNN-based facial emotion recognition model.
    Can be trained on custom dataset or use pre-trained weights.
    """
    
    def __init__(self, image_size: int = 48, n_classes: int = 7):
        self.image_size = image_size
        self.n_classes = n_classes
        self.model = None
        self.is_trained = False
        
        self.emotion_labels = [
            'angry', 'disgusted', 'fearful', 'happy',
            'neutral', 'sad', 'surprised'
        ]
        
        self._build_model()
    
    def _build_model(self):
        """Build CNN model for facial emotion"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                # Block 1
                layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                             input_shape=(self.image_size, self.image_size, 1)),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Block 2
                layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Block 3
                layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Classification head
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.n_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Facial Emotion CNN model built")
            
        except ImportError:
            print("⚠️ TensorFlow not available for Facial CNN")
            self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> Dict:
        """Train the CNN model"""
        if self.model is None:
            return {'success': False, 'error': 'Model not available'}
        
        try:
            # Data augmentation
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )
            
            history = self.model.fit(
                datagen.flow(X, y, batch_size=64),
                epochs=epochs,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'success': True,
                'final_accuracy': history.history['accuracy'][-1]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict(self, image: np.ndarray) -> Dict:
        """Predict emotion from facial image"""
        if self.model is None or not self.is_trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Preprocess image
            if image.shape[:2] != (self.image_size, self.image_size):
                import cv2
                image = cv2.resize(image, (self.image_size, self.image_size))
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image = image.reshape(1, self.image_size, self.image_size, 1) / 255.0
            
            probs = self.model.predict(image, verbose=0)
            predicted_idx = np.argmax(probs[0])
            
            return {
                'success': True,
                'emotion': self.emotion_labels[predicted_idx],
                'confidence': float(probs[0][predicted_idx]),
                'all_probabilities': {
                    label: float(probs[0][i])
                    for i, label in enumerate(self.emotion_labels)
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
