"""
Cognitive State Fusion Engine
Integrates emotional and cognitive insights from all modalities using advanced fusion techniques.
Implements attention-based multi-source fusion for emotion and state detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path


class CognitiveStateFusionEngine:
    """
    Advanced Cognitive State Fusion Engine that integrates:
    - Text sentiment analysis
    - Voice/prosody emotion detection
    - Facial expression analysis
    - Behavioral context signals
    
    Uses attention-weighted fusion and temporal smoothing for accurate state estimation.
    """
    
    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.emotion_history = deque(maxlen=history_size)
        self.cognitive_states = deque(maxlen=history_size)
        
        # Modality weights (learned dynamically)
        self.modality_weights = {
            'text': 0.30,
            'voice': 0.25,
            'facial': 0.35,
            'behavioral': 0.10
        }
        
        # Emotion embeddings for semantic similarity
        self.emotion_embeddings = {
            'happy': np.array([1.0, 0.8, 0.0, 0.0, 0.0]),
            'excited': np.array([0.9, 1.0, 0.0, 0.0, 0.0]),
            'calm': np.array([0.5, 0.2, 0.0, 0.1, 0.0]),
            'neutral': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            'sad': np.array([-0.8, -0.5, 0.0, 0.5, 0.0]),
            'angry': np.array([-0.7, 0.5, 0.8, 0.3, 0.0]),
            'fearful': np.array([-0.6, 0.2, 0.0, 0.8, 0.0]),
            'anxious': np.array([-0.3, 0.3, 0.2, 0.7, 0.0]),
            'stressed': np.array([-0.4, 0.4, 0.3, 0.6, 0.0]),
            'surprised': np.array([0.2, 0.7, 0.0, 0.3, 1.0]),
            'disgusted': np.array([-0.5, 0.1, 0.4, 0.2, 0.0]),
            'tired': np.array([-0.2, -0.7, 0.0, 0.2, 0.0])
        }
        
        # Cognitive state definitions
        self.cognitive_states_def = {
            'focused': {'energy': 'high', 'valence': 'neutral_positive', 'arousal': 'moderate'},
            'flow': {'energy': 'high', 'valence': 'positive', 'arousal': 'moderate'},
            'relaxed': {'energy': 'low', 'valence': 'positive', 'arousal': 'low'},
            'stressed': {'energy': 'high', 'valence': 'negative', 'arousal': 'high'},
            'fatigued': {'energy': 'low', 'valence': 'neutral_negative', 'arousal': 'low'},
            'distracted': {'energy': 'moderate', 'valence': 'neutral', 'arousal': 'high'},
            'productive': {'energy': 'high', 'valence': 'positive', 'arousal': 'moderate'}
        }
        
        # Load learned fusion parameters
        self.fusion_params_path = Path(__file__).parent / 'fusion_parameters.json'
        self._load_fusion_parameters()
    
    def _load_fusion_parameters(self):
        """Load learned fusion parameters from file"""
        try:
            if self.fusion_params_path.exists():
                with open(self.fusion_params_path, 'r') as f:
                    params = json.load(f)
                    self.modality_weights = params.get('modality_weights', self.modality_weights)
        except Exception as e:
            print(f"Warning: Could not load fusion parameters: {e}")
    
    def _save_fusion_parameters(self):
        """Save learned fusion parameters to file"""
        try:
            params = {
                'modality_weights': self.modality_weights,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.fusion_params_path, 'w') as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save fusion parameters: {e}")
    
    def fuse_emotion_sources(self,
                             text_emotion: Dict = None,
                             voice_emotion: Dict = None,
                             facial_emotion: Dict = None,
                             behavioral_context: Dict = None,
                             use_attention: bool = True) -> Dict:
        """
        Fuse emotions from multiple sources using attention-weighted averaging.
        
        Args:
            text_emotion: {'emotion': str, 'confidence': float, 'features': dict}
            voice_emotion: {'emotion': str, 'confidence': float, 'features': dict}
            facial_emotion: {'emotion': str, 'confidence': float, 'features': dict}
            behavioral_context: {'activity': str, 'time_of_day': str, 'recent_tasks': list}
            use_attention: Whether to use attention mechanism for weighting
            
        Returns:
            Fused cognitive state with confidence and explanation
        """
        sources = []
        source_data = []
        
        # Collect available sources
        if text_emotion and text_emotion.get('emotion'):
            sources.append('text')
            source_data.append({
                'modality': 'text',
                'emotion': text_emotion['emotion'],
                'confidence': text_emotion.get('confidence', 0.5),
                'embedding': self._get_emotion_embedding(text_emotion['emotion']),
                'valence': text_emotion.get('valence', 0.0),
                'arousal': text_emotion.get('arousal', 0.0)
            })
        
        if voice_emotion and voice_emotion.get('emotion'):
            sources.append('voice')
            source_data.append({
                'modality': 'voice',
                'emotion': voice_emotion['emotion'],
                'confidence': voice_emotion.get('confidence', 0.5),
                'embedding': self._get_emotion_embedding(voice_emotion['emotion']),
                'valence': voice_emotion.get('valence', 0.0),
                'arousal': voice_emotion.get('arousal', 0.0)
            })
        
        if facial_emotion and facial_emotion.get('emotion'):
            sources.append('facial')
            source_data.append({
                'modality': 'facial',
                'emotion': facial_emotion['emotion'],
                'confidence': facial_emotion.get('confidence', 0.5),
                'embedding': self._get_emotion_embedding(facial_emotion['emotion']),
                'valence': facial_emotion.get('valence', 0.0),
                'arousal': facial_emotion.get('arousal', 0.0)
            })
        
        if behavioral_context:
            inferred_emotion = self._infer_emotion_from_behavior(behavioral_context)
            if inferred_emotion:
                sources.append('behavioral')
                source_data.append({
                    'modality': 'behavioral',
                    'emotion': inferred_emotion['emotion'],
                    'confidence': inferred_emotion.get('confidence', 0.3),
                    'embedding': self._get_emotion_embedding(inferred_emotion['emotion']),
                    'valence': inferred_emotion.get('valence', 0.0),
                    'arousal': inferred_emotion.get('arousal', 0.0)
                })
        
        if not source_data:
            return {
                'success': False,
                'error': 'No emotion sources available',
                'primary_emotion': 'neutral',
                'confidence': 0.0
            }
        
        # Apply attention-weighted fusion
        if use_attention and len(source_data) > 1:
            fused_result = self._attention_weighted_fusion(source_data)
        else:
            fused_result = self._simple_weighted_fusion(source_data)
        
        # Determine cognitive state
        cognitive_state = self._determine_cognitive_state(
            fused_result['primary_emotion'],
            fused_result.get('arousal', 0.5),
            fused_result.get('valence', 0.0)
        )
        
        # Apply temporal smoothing
        smoothed_result = self._apply_temporal_smoothing(fused_result)
        
        # Store in history
        self.emotion_history.append({
            'timestamp': datetime.now().isoformat(),
            'result': smoothed_result,
            'sources': sources
        })
        
        return {
            'success': True,
            'primary_emotion': smoothed_result['primary_emotion'],
            'secondary_emotion': smoothed_result.get('secondary_emotion'),
            'confidence': smoothed_result['confidence'],
            'cognitive_state': cognitive_state,
            'valence': smoothed_result.get('valence', 0.0),
            'arousal': smoothed_result.get('arousal', 0.5),
            'sources_used': sources,
            'source_contributions': smoothed_result.get('contributions', {}),
            'temporal_trend': self._calculate_emotion_trend(),
            'explanation': self._generate_fusion_explanation(smoothed_result, sources)
        }
    
    def _get_emotion_embedding(self, emotion: str) -> np.ndarray:
        """Get embedding vector for an emotion"""
        return self.emotion_embeddings.get(emotion, np.zeros(5))
    
    def _attention_weighted_fusion(self, source_data: List[Dict]) -> Dict:
        """
        Apply attention mechanism to weight emotion sources based on:
        1. Source confidence scores
        2. Cross-modal agreement (attention)
        3. Historical reliability
        """
        n_sources = len(source_data)
        
        # Build attention matrix based on cross-modal agreement
        attention_scores = np.zeros((n_sources, n_sources))
        
        for i, src_i in enumerate(source_data):
            for j, src_j in enumerate(source_data):
                if i != j:
                    # Calculate similarity between embeddings
                    sim = np.dot(src_i['embedding'], src_j['embedding'])
                    sim = sim / (np.linalg.norm(src_i['embedding']) * np.linalg.norm(src_j['embedding']) + 1e-8)
                    attention_scores[i, j] = max(0, sim)
        
        # Calculate attention weights
        attention_weights = []
        for i, src in enumerate(source_data):
            # Base weight from modality prior
            base_weight = self.modality_weights.get(src['modality'], 0.25)
            
            # Confidence weight
            conf_weight = src['confidence']
            
            # Agreement weight (how much other sources agree)
            agreement = np.mean(attention_scores[i, :]) if n_sources > 1 else 1.0
            
            # Combined attention weight
            attention_weight = base_weight * conf_weight * (0.7 + 0.3 * agreement)
            attention_weights.append(attention_weight)
        
        # Normalize weights
        total_weight = sum(attention_weights)
        if total_weight > 0:
            attention_weights = [w / total_weight for w in attention_weights]
        else:
            attention_weights = [1.0 / n_sources] * n_sources
        
        # Fuse embeddings
        fused_embedding = np.zeros(5)
        for i, src in enumerate(source_data):
            fused_embedding += attention_weights[i] * src['embedding']
        
        # Find closest emotion to fused embedding
        primary_emotion, secondary_emotion = self._embedding_to_emotion(fused_embedding)
        
        # Calculate fused confidence
        fused_confidence = sum(w * src['confidence'] for w, src in zip(attention_weights, source_data))
        
        # Calculate valence and arousal
        valence = sum(w * src.get('valence', 0.0) for w, src in zip(attention_weights, source_data))
        arousal = sum(w * src.get('arousal', 0.5) for w, src in zip(attention_weights, source_data))
        
        # Record contributions for explainability
        contributions = {
            src['modality']: {
                'weight': attention_weights[i],
                'emotion': src['emotion'],
                'confidence': src['confidence']
            }
            for i, src in enumerate(source_data)
        }
        
        return {
            'primary_emotion': primary_emotion,
            'secondary_emotion': secondary_emotion,
            'confidence': fused_confidence,
            'valence': valence,
            'arousal': arousal,
            'fused_embedding': fused_embedding.tolist(),
            'contributions': contributions
        }
    
    def _simple_weighted_fusion(self, source_data: List[Dict]) -> Dict:
        """Simple weighted average fusion (fallback)"""
        fused_embedding = np.zeros(5)
        total_weight = 0
        
        for src in source_data:
            weight = self.modality_weights.get(src['modality'], 0.25) * src['confidence']
            fused_embedding += weight * src['embedding']
            total_weight += weight
        
        if total_weight > 0:
            fused_embedding /= total_weight
        
        primary_emotion, secondary_emotion = self._embedding_to_emotion(fused_embedding)
        fused_confidence = np.mean([s['confidence'] for s in source_data])
        
        return {
            'primary_emotion': primary_emotion,
            'secondary_emotion': secondary_emotion,
            'confidence': fused_confidence,
            'valence': fused_embedding[0],
            'arousal': abs(fused_embedding[1])
        }
    
    def _embedding_to_emotion(self, embedding: np.ndarray) -> Tuple[str, Optional[str]]:
        """Find the closest emotion(s) to the given embedding"""
        best_emotion = 'neutral'
        best_similarity = -1
        second_emotion = None
        second_similarity = -1
        
        for emotion, emb in self.emotion_embeddings.items():
            similarity = np.dot(embedding, emb)
            norm = np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-8
            similarity = similarity / norm
            
            if similarity > best_similarity:
                second_emotion = best_emotion
                second_similarity = best_similarity
                best_emotion = emotion
                best_similarity = similarity
            elif similarity > second_similarity:
                second_emotion = emotion
                second_similarity = similarity
        
        return best_emotion, second_emotion if second_similarity > 0.3 else None
    
    def _infer_emotion_from_behavior(self, behavioral_context: Dict) -> Optional[Dict]:
        """Infer emotion from behavioral signals"""
        if not behavioral_context:
            return None
        
        activity = behavioral_context.get('activity', '')
        time_of_day = behavioral_context.get('time_of_day', '')
        recent_tasks = behavioral_context.get('recent_tasks', [])
        task_completion_rate = behavioral_context.get('task_completion_rate', 0.5)
        
        # Simple heuristic-based inference
        if task_completion_rate > 0.8:
            return {'emotion': 'happy', 'confidence': 0.4, 'valence': 0.5, 'arousal': 0.5}
        elif task_completion_rate < 0.3:
            return {'emotion': 'stressed', 'confidence': 0.4, 'valence': -0.3, 'arousal': 0.6}
        elif activity == 'working' and time_of_day == 'morning':
            return {'emotion': 'calm', 'confidence': 0.3, 'valence': 0.2, 'arousal': 0.4}
        
        return {'emotion': 'neutral', 'confidence': 0.2, 'valence': 0.0, 'arousal': 0.5}
    
    def _determine_cognitive_state(self, emotion: str, arousal: float, valence: float) -> str:
        """Determine cognitive state from emotion and physiological signals"""
        # Simple rule-based determination
        if valence > 0.3 and arousal > 0.5:
            return 'productive' if emotion in ['happy', 'excited'] else 'focused'
        elif valence > 0.3 and arousal <= 0.5:
            return 'relaxed'
        elif valence < -0.3 and arousal > 0.5:
            return 'stressed'
        elif valence < -0.3 and arousal <= 0.5:
            return 'fatigued'
        elif arousal > 0.7:
            return 'distracted'
        
        return 'focused'
    
    def _apply_temporal_smoothing(self, current_result: Dict, alpha: float = 0.7) -> Dict:
        """
        Apply exponential smoothing to reduce noise in emotion detection.
        Uses weighted combination of current and historical states.
        """
        if len(self.emotion_history) < 2:
            return current_result
        
        # Get recent history
        recent = list(self.emotion_history)[-5:]
        
        # Calculate emotion frequency
        emotion_counts = {}
        for entry in recent:
            emotion = entry['result'].get('primary_emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # If current emotion is consistent with history, increase confidence
        current_emotion = current_result['primary_emotion']
        history_ratio = emotion_counts.get(current_emotion, 0) / len(recent)
        
        smoothed_confidence = alpha * current_result['confidence'] + (1 - alpha) * history_ratio
        
        result = current_result.copy()
        result['confidence'] = min(1.0, smoothed_confidence)
        result['smoothing_applied'] = True
        
        return result
    
    def _calculate_emotion_trend(self) -> Dict:
        """Calculate emotional trend over recent history"""
        if len(self.emotion_history) < 3:
            return {'trend': 'stable', 'direction': 'neutral'}
        
        recent = list(self.emotion_history)[-10:]
        
        # Calculate valence trend
        valences = [
            entry['result'].get('valence', 0.0) 
            for entry in recent 
            if 'result' in entry
        ]
        
        if len(valences) >= 3:
            # Simple linear regression
            x = np.arange(len(valences))
            slope = np.polyfit(x, valences, 1)[0]
            
            if slope > 0.05:
                return {'trend': 'improving', 'direction': 'positive', 'slope': float(slope)}
            elif slope < -0.05:
                return {'trend': 'declining', 'direction': 'negative', 'slope': float(slope)}
        
        return {'trend': 'stable', 'direction': 'neutral', 'slope': 0.0}
    
    def _generate_fusion_explanation(self, result: Dict, sources: List[str]) -> str:
        """Generate human-readable explanation of fusion result"""
        contributions = result.get('contributions', {})
        
        explanation_parts = [f"Detected {result['primary_emotion']} emotion"]
        
        if contributions:
            top_source = max(contributions.items(), key=lambda x: x[1]['weight'])
            explanation_parts.append(
                f"primarily based on {top_source[0]} analysis "
                f"({top_source[1]['emotion']}, {top_source[1]['weight']:.0%} contribution)"
            )
        
        if result.get('secondary_emotion'):
            explanation_parts.append(f"with underlying {result['secondary_emotion']}")
        
        return ' '.join(explanation_parts)
    
    def update_modality_weights(self, feedback: Dict):
        """
        Update modality weights based on user feedback for adaptive learning.
        
        Args:
            feedback: {'correct_emotion': str, 'predicted_emotion': str, 'sources': dict}
        """
        if not feedback:
            return
        
        correct = feedback.get('correct_emotion')
        predicted = feedback.get('predicted_emotion')
        sources = feedback.get('sources', {})
        
        if correct == predicted:
            # Increase weights for sources that predicted correctly
            for modality, data in sources.items():
                if data['emotion'] == correct:
                    current_weight = self.modality_weights.get(modality, 0.25)
                    self.modality_weights[modality] = min(0.5, current_weight * 1.05)
        else:
            # Decrease weights for sources that predicted incorrectly
            for modality, data in sources.items():
                if data['emotion'] != correct:
                    current_weight = self.modality_weights.get(modality, 0.25)
                    self.modality_weights[modality] = max(0.1, current_weight * 0.95)
        
        # Normalize weights
        total = sum(self.modality_weights.values())
        self.modality_weights = {k: v/total for k, v in self.modality_weights.items()}
        
        # Save updated parameters
        self._save_fusion_parameters()
    
    def get_productivity_estimate(self) -> Dict:
        """
        Estimate user productivity based on cognitive state history.
        
        Returns:
            Dict with productivity score, energy level, and recommendations
        """
        if len(self.emotion_history) < 3:
            return {
                'productivity_score': 0.5,
                'energy_level': 'moderate',
                'insights': 'Insufficient data for accurate estimation',
                'recommendation': 'Continue interacting for personalized insights'
            }
        
        recent = list(self.emotion_history)[-20:]
        
        # Calculate positive emotion ratio
        positive_emotions = ['happy', 'excited', 'calm']
        negative_emotions = ['sad', 'angry', 'stressed', 'anxious', 'tired']
        
        positive_count = sum(
            1 for entry in recent 
            if entry['result'].get('primary_emotion') in positive_emotions
        )
        negative_count = sum(
            1 for entry in recent 
            if entry['result'].get('primary_emotion') in negative_emotions
        )
        
        total = len(recent)
        positive_ratio = positive_count / total if total > 0 else 0.5
        negative_ratio = negative_count / total if total > 0 else 0.3
        
        # Calculate productivity score
        productivity_score = 0.5 + (positive_ratio - negative_ratio) * 0.5
        productivity_score = max(0.0, min(1.0, productivity_score))
        
        # Determine energy level
        avg_arousal = np.mean([
            entry['result'].get('arousal', 0.5) 
            for entry in recent 
            if 'result' in entry
        ])
        
        if avg_arousal > 0.6:
            energy_level = 'high'
        elif avg_arousal > 0.4:
            energy_level = 'moderate'
        else:
            energy_level = 'low'
        
        # Generate recommendation
        if productivity_score > 0.7:
            recommendation = "Great energy! Good time for complex tasks."
        elif productivity_score < 0.4:
            recommendation = "Consider taking a break or trying a simpler task."
        else:
            recommendation = "Maintain pace with moderately challenging work."
        
        return {
            'productivity_score': productivity_score,
            'energy_level': energy_level,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'avg_arousal': float(avg_arousal),
            'recommendation': recommendation,
            'insights': f"Based on {total} recent emotional readings"
        }
    
    def get_stress_level(self) -> Dict:
        """
        Calculate current stress level from cognitive state history.
        
        Returns:
            Dict with stress score, factors, and mitigation suggestions
        """
        if len(self.emotion_history) < 3:
            return {
                'stress_score': 0.3,
                'level': 'low',
                'contributing_factors': [],
                'suggestions': ['Insufficient data - continue using the assistant']
            }
        
        recent = list(self.emotion_history)[-15:]
        
        # Stress indicators
        stress_emotions = ['stressed', 'anxious', 'angry', 'fearful']
        stress_count = sum(
            1 for entry in recent 
            if entry['result'].get('primary_emotion') in stress_emotions
        )
        
        # Emotional volatility (changes in emotion)
        emotions = [entry['result'].get('primary_emotion', 'neutral') for entry in recent]
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        volatility = changes / len(emotions) if emotions else 0
        
        # Calculate stress score
        stress_ratio = stress_count / len(recent) if recent else 0
        stress_score = 0.6 * stress_ratio + 0.4 * volatility
        stress_score = max(0.0, min(1.0, stress_score))
        
        # Determine level
        if stress_score > 0.7:
            level = 'high'
            suggestions = [
                "Consider taking a 15-minute break",
                "Try deep breathing exercises",
                "Postpone non-urgent tasks if possible"
            ]
        elif stress_score > 0.4:
            level = 'moderate'
            suggestions = [
                "Brief pause recommended",
                "Prioritize tasks to reduce overwhelm",
                "Consider lighter activities"
            ]
        else:
            level = 'low'
            suggestions = [
                "Maintain current pace",
                "Good time for challenging tasks"
            ]
        
        contributing_factors = []
        if stress_ratio > 0.3:
            contributing_factors.append("High frequency of stress-related emotions")
        if volatility > 0.5:
            contributing_factors.append("High emotional variability")
        
        return {
            'stress_score': stress_score,
            'level': level,
            'contributing_factors': contributing_factors,
            'suggestions': suggestions,
            'stress_emotion_ratio': stress_ratio,
            'emotional_volatility': volatility
        }


class RealTimeCognitiveMonitor:
    """
    Real-time cognitive state monitoring with continuous updates.
    Designed for WebSocket integration and live dashboard updates.
    """
    
    def __init__(self, fusion_engine: Optional[CognitiveStateFusionEngine] = None):
        self.fusion_engine = fusion_engine or CognitiveStateFusionEngine()
        self.monitoring_active = False
        self.update_callbacks = []
        self.latest_state = None
    
    def register_callback(self, callback):
        """Register a callback for state updates"""
        self.update_callbacks.append(callback)
    
    def update_state(self,
                     text_input: str = None,
                     voice_features: Dict = None,
                     facial_features: Dict = None,
                     behavioral_data: Dict = None) -> Dict:
        """
        Update cognitive state with new input data.
        
        This method should be called whenever new data is available
        from any modality (text, voice, facial, behavioral).
        """
        from .nlp_engine import NLPEngine
        from .emotion_analyzer import EmotionAnalyzer
        
        text_emotion = None
        voice_emotion = None
        facial_emotion = None
        
        # Process text input
        if text_input:
            nlp = NLPEngine()
            text_result = nlp.detect_emotion_from_text(text_input)
            text_emotion = {
                'emotion': text_result['emotion'],
                'confidence': text_result['confidence'],
                'valence': text_result['sentiment_scores'].get('compound', 0.0),
                'arousal': abs(text_result.get('intensity', 0.5))
            }
        
        # Process voice features
        if voice_features:
            voice_emotion = self._process_voice_features(voice_features)
        
        # Process facial features
        if facial_features:
            facial_emotion = {
                'emotion': facial_features.get('emotion', 'neutral'),
                'confidence': facial_features.get('confidence', 0.5),
                'valence': facial_features.get('valence', 0.0),
                'arousal': facial_features.get('arousal', 0.5)
            }
        
        # Fuse all sources
        result = self.fusion_engine.fuse_emotion_sources(
            text_emotion=text_emotion,
            voice_emotion=voice_emotion,
            facial_emotion=facial_emotion,
            behavioral_context=behavioral_data
        )
        
        self.latest_state = result
        
        # Notify callbacks
        for callback in self.update_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Callback error: {e}")
        
        return result
    
    def _process_voice_features(self, features: Dict) -> Dict:
        """Process raw voice features into emotion prediction"""
        # Simple feature-based emotion inference
        pitch = features.get('pitch', 0.5)
        energy = features.get('energy', 0.5)
        speech_rate = features.get('speech_rate', 0.5)
        
        # Heuristic emotion inference
        if energy > 0.7 and pitch > 0.6:
            emotion = 'excited'
            valence = 0.6
        elif energy < 0.3 and pitch < 0.4:
            emotion = 'sad'
            valence = -0.5
        elif energy > 0.6 and speech_rate > 0.7:
            emotion = 'anxious'
            valence = -0.2
        else:
            emotion = 'neutral'
            valence = 0.0
        
        return {
            'emotion': emotion,
            'confidence': 0.4,  # Lower confidence for simple inference
            'valence': valence,
            'arousal': energy
        }
    
    def get_current_state(self) -> Dict:
        """Get the latest cognitive state"""
        return self.latest_state or {
            'primary_emotion': 'neutral',
            'cognitive_state': 'focused',
            'confidence': 0.0
        }
    
    def get_analytics(self) -> Dict:
        """Get cognitive analytics summary"""
        return {
            'productivity': self.fusion_engine.get_productivity_estimate(),
            'stress': self.fusion_engine.get_stress_level(),
            'emotion_trend': self.fusion_engine._calculate_emotion_trend(),
            'history_size': len(self.fusion_engine.emotion_history)
        }
