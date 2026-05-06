"""
Unified AI Services Integration API
Provides a unified interface to all cognitive AI services.
Enables seamless integration between components and REST API exposure.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
from threading import Lock

# Import all AI service modules
from .cognitive_fusion_engine import CognitiveStateFusionEngine, RealTimeCognitiveMonitor
from .behavioral_sequence_modeler import BehavioralSequenceModeler, BehaviorPatternMiner
from .multimodal_emotion_detector import MultimodalEmotionDetector, RealTimeEmotionTracker
from .rl_task_scheduler import RLTaskScheduler, ContextAwareScheduler
from .explainable_ai_layer import ExplainableAILayer, TaskRecommendationExplainer, EmotionExplainer
from .adaptive_personality_engine import AdaptivePersonalityEngine, PersonalityEvolutionTracker
from .habit_formation_predictor import HabitFormationPredictor
from .anomaly_detection_system import AnomalyDetectionSystem
from .cognitive_analytics_dashboard import CognitiveAnalyticsDashboard


_SERVICE_CACHE: Dict[str, "CognitiveAIServices"] = {}
_SERVICE_CACHE_LOCK = Lock()


class CognitiveAIServices:
    """
    Unified interface for all cognitive AI services.
    Manages service lifecycle, orchestration, and inter-service communication.
    """

    _shared_component_lock = Lock()
    _shared_emotion_detector: Optional[MultimodalEmotionDetector] = None
    _shared_xai_layer: Optional[ExplainableAILayer] = None
    
    def __init__(self, user_id: str):
        """
        Initialize all AI services for a user.
        
        Args:
            user_id: User identifier for personalization
        """
        self.user_id = user_id
        self._services_initialized = False
        
        # Initialize attributes to None for safety
        self.cognitive_fusion = None
        self.cognitive_monitor = None
        self.behavior_modeler = None
        self.pattern_miner = None
        self.emotion_detector = None
        self.emotion_tracker = None
        self.rl_scheduler = None
        self.context_scheduler = None
        self.xai_layer = None
        self.task_explainer = None
        self.emotion_explainer = None
        self.personality_engine = None
        self.personality_tracker = None
        self.habit_predictor = None
        self.anomaly_detector = None
        self.analytics_dashboard = None
        
        # Initialize all services
        self._init_services()
    
    def _init_services(self):
        """Initialize all AI service components"""
        try:
            # Core engines
            self.cognitive_fusion = CognitiveStateFusionEngine()
            self.cognitive_monitor = RealTimeCognitiveMonitor(self.cognitive_fusion)
            
            # Behavior modeling
            self.behavior_modeler = BehavioralSequenceModeler(self.user_id)
            self.pattern_miner = BehaviorPatternMiner()
            
            # Emotion detection
            self.emotion_detector = self._get_shared_emotion_detector()
            self.emotion_tracker = RealTimeEmotionTracker(self.emotion_detector)
            
            # Task scheduling
            self.rl_scheduler = RLTaskScheduler(user_id=self.user_id)
            self.context_scheduler = ContextAwareScheduler(rl_scheduler=self.rl_scheduler, user_id=self.user_id)
            
            # Explainability
            self.xai_layer = self._get_shared_xai_layer()
            self.task_explainer = TaskRecommendationExplainer(self.xai_layer)
            self.emotion_explainer = EmotionExplainer(self.xai_layer)
            
            # Personality
            self.personality_engine = AdaptivePersonalityEngine(self.user_id)
            self.personality_tracker = PersonalityEvolutionTracker(self.personality_engine)
            
            # Habit tracking
            self.habit_predictor = HabitFormationPredictor(self.user_id)
            
            # Anomaly detection
            self.anomaly_detector = AnomalyDetectionSystem(self.user_id)
            
            # Analytics
            self.analytics_dashboard = CognitiveAnalyticsDashboard(self.user_id)
            
            # Register cross-service callbacks
            self._setup_service_callbacks()
            
            self._services_initialized = True
            
        except Exception as e:
            print(f"Warning: Some services failed to initialize: {e}")
            self._services_initialized = False

    @classmethod
    def _get_shared_emotion_detector(cls) -> MultimodalEmotionDetector:
        """Reuse one detector instance so transformer/deepface models are loaded once."""
        with cls._shared_component_lock:
            if cls._shared_emotion_detector is None:
                cls._shared_emotion_detector = MultimodalEmotionDetector()
            return cls._shared_emotion_detector

    @classmethod
    def _get_shared_xai_layer(cls) -> ExplainableAILayer:
        """Reuse one XAI layer instance to avoid repeated SHAP/LIME imports."""
        with cls._shared_component_lock:
            if cls._shared_xai_layer is None:
                cls._shared_xai_layer = ExplainableAILayer()
            return cls._shared_xai_layer
    
    def _setup_service_callbacks(self):
        """Set up callbacks between services"""
        # Anomaly alerts feed into analytics
        def anomaly_to_analytics(anomaly: Dict):
            # Log anomaly to analytics
            pass
        
        self.anomaly_detector.register_alert_callback(anomaly_to_analytics)
    
    # ==================== Emotion Services ====================
    
    def persist_emotion_state(self, emotion_result: Dict, source: str = 'text', text_content: str = None) -> Any:
        """
        Save emotion detection result to database for dashboard visibility.
        """
        try:
            from apps.emotion_tracker.models import EmotionState
            from django.contrib.auth import get_user_model
            User = get_user_model()
            user = User.objects.get(id=self.user_id)
            
            # Map fusion result or direct result
            emotion = emotion_result.get('primary_emotion') or emotion_result.get('emotion', 'neutral')
            confidence = emotion_result.get('confidence', 0.5)
            intensity = emotion_result.get('intensity', confidence)
            
            state = EmotionState.objects.create(
                user=user,
                emotion=emotion,
                intensity=intensity,
                source=source,
                confidence_score=confidence,
                text_content=text_content
            )
            return state
        except Exception as e:
            print(f"Error persisting emotion state: {e}")
            return None

    def analyze_emotion(self, 
                       text: str = None,
                       audio_data: Any = None,
                       image_data: Any = None) -> Dict:
        """
        Analyze emotion from multiple modalities.
        
        Args:
            text: Text input
            audio_data: Audio data (numpy array or bytes)
            image_data: Image data (numpy array or bytes)
            
        Returns:
            Comprehensive emotion analysis
        """
        if not self.emotion_detector:
            return {
                'success': False,
                'error': 'Emotion detector inactive',
                'fused_emotion': {'emotion': 'neutral', 'confidence': 0.0}
            }
            
        result = self.emotion_detector.detect_multimodal_emotion(
            text=text,
            audio_data=audio_data,
            image_data=image_data
        )
        
        # Record for tracking
        if result.get('fused_emotion'):
            self.emotion_tracker.start_tracking()
            
            # Feed to anomaly detection
            self.anomaly_detector.record_emotion_state(
                emotion=result['fused_emotion']['emotion'],
                valence=result['fused_emotion']['valence'],
                arousal=result['fused_emotion']['arousal'],
                confidence=result['fused_emotion']['confidence'],
                source='multimodal'
            )
            
            # Feed to analytics
            self.analytics_dashboard.ingest_emotion_data({
                'emotion': result['fused_emotion']['emotion'],
                'valence': result['fused_emotion']['valence'],
                'arousal': result['fused_emotion']['arousal'],
                'confidence': result['fused_emotion']['confidence'],
                'source': 'multimodal'
            })
            
            # Get explanation
            explanation = self.emotion_explainer.explain_emotion_detection(result)
            result['explanation'] = explanation
        
        return result
    
    def get_cognitive_state(self) -> Dict:
        """Get current cognitive state from all sources"""
        if not self.cognitive_monitor or not self.cognitive_monitor.latest_state:
            # Try to calculate from recent history to make it more "alive"
            from apps.emotion_tracker.models import EmotionState
            recent = EmotionState.objects.filter(user_id=self.user_id).order_by('-detected_at')[:5]
            
            if recent.exists():
                latest = recent[0]
                avg_intensity = sum(e.intensity for e in recent) / recent.count()
                
                # Map intensity and emotion to metrics
                energy = min(max(0.4, avg_intensity + 0.2 if latest.emotion in ['excited', 'happy'] else avg_intensity), 0.9)
                stress = 0.6 if latest.emotion in ['stressed', 'anxious', 'angry'] else 0.2
                focus = 0.8 if latest.emotion in ['calm', 'neutral', 'happy'] else 0.5
                
                return {
                    'primary_emotion': latest.emotion,
                    'cognitive_state': 'productive' if energy > 0.6 else 'focused',
                    'energy_level': energy,
                    'focus_level': focus,
                    'stress_level': stress,
                    'confidence': latest.confidence_score
                }

            # Ultimate fallback if no data
            return {
                'primary_emotion': 'neutral',
                'cognitive_state': 'focused',
                'energy_level': 0.75,
                'focus_level': 0.85,
                'stress_level': 0.15,
                'confidence': 0.9
            }
        
        # Get raw state from monitor
        raw_state = self.cognitive_monitor.get_current_state()
        
        # Map raw fields to expected dashboard fields if they don't exist
        if 'energy_level' not in raw_state:
            # Map arousal to energy (0-1)
            raw_state['energy_level'] = raw_state.get('arousal', 0.5)
            
        if 'focus_level' not in raw_state:
            # Map confidence/valence to focus
            valence = raw_state.get('valence', 0.0)
            confidence = raw_state.get('confidence', 0.5)
            # High focus usually correlates with high confidence and neutral/positive valence
            raw_state['focus_level'] = min(max(0.0, confidence * (1.0 + valence * 0.2)), 1.0)
            
        if 'stress_level' not in raw_state:
            # map negative valence to stress
            valence = raw_state.get('valence', 0.0)
            arousal = raw_state.get('arousal', 0.5)
            # Stress is often high arousal + negative valence
            stress_base = abs(min(0, valence)) 
            raw_state['stress_level'] = min(max(0.0, stress_base * 0.7 + arousal * 0.3), 1.0)
            
        return raw_state
    
    # ==================== Behavior Services ====================
    
    def record_activity(self, 
                       activity_type: str,
                       category: str = 'general',
                       duration: float = None,
                       metadata: Dict = None) -> Dict:
        """
        Record user activity for behavior modeling.
        
        Args:
            activity_type: Type of activity
            category: Activity category
            duration: Duration in minutes
            metadata: Additional metadata
            
        Returns:
            Recording result with predictions
        """
        now = datetime.now()
        
        # Encode activity
        activity_encoded = self.behavior_modeler._encode_activity({
            'activity_type': activity_type,
            'category': category,
            'hour': now.hour,
            'day_of_week': now.weekday()
        })
        
        # Record in behavior tracker
        self.behavior_modeler.record_activity(activity_encoded)
        
        # Feed to anomaly detection
        anomaly_result = self.anomaly_detector.record_behavior_event(
            event_type=activity_type,
            category=category,
            duration=duration,
            metadata=metadata
        )
        
        # Feed to analytics
        self.analytics_dashboard.ingest_productivity_data({
            'productivity_score': 0.7,  # Could be enhanced
            'focus_time_minutes': duration or 0,
            'tasks_completed': 1 if activity_type == 'task_complete' else 0,
            'context': category
        })
        
        # Get predictions
        predictions = self.behavior_modeler.predict_next_activities()
        
        return {
            'recorded': True,
            'activity': activity_type,
            'category': category,
            'predictions': predictions,
            'anomalies': anomaly_result.get('anomalies', [])
        }
    
    def get_behavior_patterns(self) -> Dict:
        """Get detected behavior patterns"""
        routines = self.behavior_modeler.detect_routines()
        return {
            'routines': routines,
            'model_info': {
                'sequence_length': len(self.behavior_modeler.activity_sequence),
                'patterns_detected': len(routines)
            }
        }
    
    # ==================== Task Scheduling Services ====================
    
    def get_task_recommendations(self,
                                tasks: List[Dict],
                                current_state: Dict = None) -> Dict:
        """
        Get AI-powered task scheduling recommendations.
        
        Args:
            tasks: List of tasks to schedule
            current_state: Current user state (emotion, energy, etc.)
            
        Returns:
            Prioritized task recommendations with explanations
        """
        # Get current state if not provided
        if not current_state:
            current_state = {
                'energy_level': 0.7,
                'stress_level': 0.3,
                'focus_hours_today': 2,
                'hour_of_day': datetime.now().hour
            }
        
        # Get scheduling from RL scheduler
        if not self.context_scheduler:
            return {
                'recommended_order': tasks,
                'explanations': ["AI Scheduler inactive. Using default order."],
                'context_considered': current_state
            }
            
        scheduled = self.context_scheduler.batch_schedule(tasks, current_state)
        
        # Flatten and sort for UI
        flattened_recommendations = []
        for item in scheduled:
            task = item['task']
            schedule = item['schedule']
            # Merge task data with schedule data for flat access in template
            flat_rec = task.copy()
            flat_rec.update({
                'recommended_hour': schedule.get('recommended_hour'),
                'time_range': schedule.get('time_range'),
                'score': schedule.get('score', 0),
                'explanation': schedule.get('explanation', '')
            })
            flattened_recommendations.append(flat_rec)
        
        # Sort by score descending (best recommendations first)
        flattened_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Get explanations for recommendations
        explanations = []
        if self.task_explainer:
            for item in scheduled[:5]:  # Use original scheduled list for explainer
                task_details = item['task']
                rec_result = item['schedule']
                
                explanation = self.task_explainer.explain_schedule(
                    task=task_details,
                    recommended_time=rec_result.get('recommended_hour', 0),
                    user_state=current_state
                )
                explanations.append(explanation)
        else:
            explanations = ["Reasoning module inactive."]
        
        return {
            'recommended_order': flattened_recommendations,
            'explanations': explanations,
            'context_considered': current_state
        }
    
    def record_task_completion(self,
                              task_id: str,
                              task: Dict,
                              completion_quality: float = 0.8,
                              actual_duration: float = None) -> Dict:
        """
        Record task completion for learning.
        
        Args:
            task_id: Task identifier
            task: Task details
            completion_quality: Quality of completion (0-1)
            actual_duration: Actual time taken
            
        Returns:
            Learning update results
        """
        # Calculate reward for RL
        reward = completion_quality
        
        # Update RL scheduler
        if self.rl_scheduler and hasattr(self.rl_scheduler, 'learn'):
            self.rl_scheduler.learn(reward=reward)
        
        # Record in analytics
        if self.analytics_dashboard:
            self.analytics_dashboard.ingest_task_data({
                'task_id': task_id,
                'task_name': task.get('title', task.get('name')),
                'priority': task.get('priority', 'medium'),
                'category': task.get('category', 'general'),
                'estimated_duration': task.get('estimated_duration'),
                'actual_duration': actual_duration,
                'completed': True,
                'completion_quality': completion_quality
            })
        
        return {
            'recorded': True,
            'task_id': task_id,
            'reward_given': reward,
            'scheduler_updated': self.rl_scheduler is not None
        }
    
    # ==================== Habit Services ====================
    
    def create_habit(self,
                    habit_name: str,
                    category: str = 'general',
                    frequency: str = 'daily',
                    difficulty: float = 0.5,
                    cue: str = None,
                    reward: str = None) -> Dict:
        """
        Create a new habit to track.
        
        Args:
            habit_name: Name of the habit
            category: Habit category
            frequency: How often (daily, weekly, etc.)
            difficulty: Perceived difficulty (0-1)
            cue: Trigger cue
            reward: Reward after completion
            
        Returns:
            Habit creation result with predictions
        """
        if not self.habit_predictor:
            return {'success': False, 'error': 'Habit predictor inactive'}
            
        return self.habit_predictor.create_habit(
            habit_name=habit_name,
            category=category,
            target_frequency=frequency,
            difficulty=difficulty,
            cue=cue,
            reward=reward
        )
    
    def log_habit_completion(self,
                            habit_id: str,
                            completed: bool = True,
                            difficulty_felt: float = None) -> Dict:
        """
        Log habit completion.
        
        Args:
            habit_id: Habit identifier
            completed: Whether completed
            difficulty_felt: Subjective difficulty
            
        Returns:
            Updated stats and predictions
        """
        if not self.habit_predictor:
            return {'success': False, 'error': 'Habit predictor inactive'}
            
        result = self.habit_predictor.log_completion(
            habit_id=habit_id,
            completed=completed,
            difficulty_felt=difficulty_felt
        )
        
        # Feed to analytics
        if result.get('success') and self.analytics_dashboard:
            habit = self.habit_predictor.habits.get(habit_id, {})
            self.analytics_dashboard.ingest_habit_data({
                'habit_id': habit_id,
                'habit_name': habit.get('name', habit_id),
                'completed': completed,
                'streak': habit.get('current_streak', 0),
                'difficulty': difficulty_felt or habit.get('difficulty', 0.5)
            })
        
        return result
    
    def get_daily_habits(self) -> List[Dict]:
        """Get today's habit schedule"""
        if not self.habit_predictor:
            return []
        return self.habit_predictor.get_daily_habit_schedule()
    
    # ==================== Conversation Services ====================
    
    def process_conversation(self,
                            user_message: str,
                            assistant_response: str) -> Dict:
        """
        Process a conversation turn for learning.
        
        Args:
            user_message: What the user said
            assistant_response: What the assistant replied
            
        Returns:
            Interaction analysis and adaptations
        """
        # Get current emotion
        if not self.emotion_detector:
            emotion_state = 'neutral'
        else:
            emotion_result = self.emotion_detector.detect_multimodal_emotion(text=user_message)
            emotion_state = emotion_result.get('fused_emotion', {}).get('emotion', 'neutral')
        
        # Record for personality learning
        result = {}
        if self.personality_engine:
            result = self.personality_engine.record_interaction(
                user_message=user_message,
                assistant_response=assistant_response,
                emotion_state=emotion_state
            )
        
        # Record for anomaly detection
        if self.anomaly_detector:
            self.anomaly_detector.record_interaction(
                message_length=len(user_message),
                response_time=0.5,  # Could be actual
                sentiment=emotion_result.get('text_emotion', {}).get('valence', 0),
                intent='general'
            )
        
        return result or {'success': True, 'note': 'Service adaptation in background'}
    
    def get_response_style(self, context: Dict = None) -> Dict:
        """
        Get recommended response style for current context.
        """
        if not self.personality_engine:
            return {'tone': 'friendly', 'style': 'helpful'}
        return self.personality_engine.get_response_style(context)
    
    def get_system_prompt(self, context: Dict = None) -> str:
        """
        Get LLM system prompt reflecting current personality.
        """
        if not self.personality_engine:
            return "You are a helpful AI assistant."
        return self.personality_engine.generate_system_prompt(context)
    
    def generate_response(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """
        Generate AI response using NLPEngine with current cognitive context.
        """
        from .nlp_engine import NLPEngine
        nlp = NLPEngine()
        
        # Get enriched context
        context = self.get_enriched_chat_context()
        
        # Extract user context for NLPEngine
        user_context = {
            'name': context.get('user_name', 'User'),
            'current_emotion': context.get('emotion_state', {}).get('current_emotion', 'neutral')
        }
        
        # Generate response through the hybrid pipeline.
        response_data = nlp.generate_smart_response(
            user_message=user_message,
            conversation_history=conversation_history or [],
            user_context=user_context,
            system_prompt=context.get('system_prompt')
        )

        return response_data.get('response', "I am here to help. Please try again in a moment.")

    # ==================== Analytics Services ====================
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard summary"""
        if not self.analytics_dashboard:
            return {'summary': 'Analytics service unavailable'}
        return self.analytics_dashboard.get_dashboard_summary()
    
    def get_emotion_trends(self, period: str = 'week') -> Dict:
        """Get emotion trends over time"""
        if not self.analytics_dashboard: return {}
        return self.analytics_dashboard.get_emotion_trends(period=period)
    
    def get_productivity_trends(self, period: str = 'week') -> Dict:
        """Get productivity trends over time"""
        if not self.analytics_dashboard: return {}
        return self.analytics_dashboard.get_productivity_trends(period=period)
    
    def get_analytics_report(self, period: str = 'month') -> Dict:
        """Get comprehensive analytics report"""
        if not self.analytics_dashboard: return {}
        return self.analytics_dashboard.export_analytics_report(period=period)
    
    def get_correlation_analysis(self, period: str = 'month') -> Dict:
        """Get emotion-productivity correlation analysis"""
        if not self.analytics_dashboard: return {}
        return self.analytics_dashboard.get_emotion_productivity_correlation(period=period)
    
    # ==================== Anomaly Services ====================
    
    def get_risk_assessment(self) -> Dict:
        """Get current risk assessment"""
        if not self.anomaly_detector: return {'risk_level': 'low', 'note': 'Service inactive'}
        return self.anomaly_detector.get_current_risk_assessment()
    
    def get_anomaly_summary(self, days: int = 7) -> Dict:
        """Get anomaly summary"""
        if not self.anomaly_detector: return {}
        return self.anomaly_detector.get_anomaly_summary(days=days)
    
    def get_wellness_insights(self) -> Dict:
        """Get wellness insights based on patterns"""
        if not self.anomaly_detector: return {'insights': []}
        return self.anomaly_detector.get_wellness_insights()
    
    # ==================== Explainability Services ====================
    
    def explain_recommendation(self, 
                              recommendation_type: str,
                              recommendation: Dict,
                              context: Dict = None) -> Dict:
        """
        Get explanation for any AI recommendation.
        
        Args:
            recommendation_type: Type (task, emotion, habit, etc.)
            recommendation: The recommendation to explain
            context: Context in which it was made
            
        Returns:
            Human-readable explanation
        """
        if recommendation_type == 'task':
            if not self.task_explainer: return {'explanation': 'Explainer unavailable'}
            return self.task_explainer.explain_schedule(
                task=recommendation,
                recommended_time=recommendation.get('recommended_hour', 0),
                user_state=context or {}
            )
        elif recommendation_type == 'emotion':
            if not self.emotion_explainer: return {'explanation': 'Explainer unavailable'}
            return self.emotion_explainer.explain_emotion(
                detected_emotion=recommendation.get('emotion', 'unknown'),
                modality_results=recommendation.get('modalities', {}),
                confidence=recommendation.get('confidence', 0.0)
            )
        else:
            if not self.xai_layer: return {'explanation': 'XAI layer unavailable'}
            return self.xai_layer.generate_natural_language_explanation(
                decision_type=recommendation_type,
                decision=recommendation,
                feature_importances={},
                context=context or {}
            )
    
    # ==================== Model Management ====================
    
    def train_models(self) -> Dict:
        """Train all trainable models"""
        results = {}
        
        # Train anomaly detection
        if self.anomaly_detector:
            results['anomaly_detection'] = self.anomaly_detector.train_anomaly_models()
        
        # Train behavior models
        if self.behavior_modeler and hasattr(self.behavior_modeler, 'train_lstm'):
            results['behavior_lstm'] = self.behavior_modeler.train_lstm()
        
        # Train habit predictor
        if self.habit_predictor:
            results['habit_predictor'] = self.habit_predictor.train_models()
        
        return results
    
    def get_emotional_state(self) -> Dict:
        """Get current emotional state from analytics and monitor"""
        cognitive = self.get_cognitive_state()
        
        # If we have analytics data, use the latest entry
        if self.analytics_dashboard and self.analytics_dashboard.emotion_data:
            latest = self.analytics_dashboard.emotion_data[-1]
            return {
                'current_emotion': latest.get('emotion', cognitive.get('primary_emotion', 'neutral')),
                'avg_valence': latest.get('valence', 0.5),
                'avg_arousal': latest.get('arousal', 0.5),
                'confidence': latest.get('confidence', 0.8)
            }
        
        # Fallback to cognitive state
        return {
            'current_emotion': cognitive.get('primary_emotion', 'neutral'),
            'avg_valence': 0.5,
            'avg_arousal': 0.5,
            'confidence': cognitive.get('confidence', 0.0)
        }

    # ==================== Unified Context ====================
    
    def get_enriched_chat_context(self) -> Dict:
        """
        Generates an enriched context for the chat system.
        Combines personality, cognitive state, and emotional data for LLM prompts.
        """
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user_name = "User"
        try:
            user = User.objects.get(id=self.user_id)
            user_name = user.first_name or user.username
        except:
            pass
            
        cognitive_state = self.get_cognitive_state()
        emotion_state = self.get_emotional_state()
        personality_prompt = ""
        
        if self.personality_engine:
            personality_prompt = self.personality_engine.generate_system_prompt({
                'user_name': user_name,
                'user_emotion': emotion_state.get('current_emotion', 'neutral'),
                'cognitive_metrics': cognitive_state
            })
            
        # Create a self-reflection analytical prompt
        analytical_appendix = (
            f"\n\n[ANALYTICAL CONTEXT (DO NOT STATE DIRECTLY BUT USE IN REASONING)]\n"
            f"User Cognitive State: Energy={cognitive_state.get('energy_level', cognitive_state.get('energy', 0.5))}, "
            f"Focus={cognitive_state.get('focus_level', cognitive_state.get('focus', 0.5))}, "
            f"Stress={cognitive_state.get('stress_level', cognitive_state.get('stress', 0.5))}.\n"
            f"Detected Mood: {emotion_state.get('current_emotion', 'neutral')} (Valence: {emotion_state.get('avg_valence', 0.5)}).\n"
            f"Analysis Strategy: Provide deep, analytical answers. For project queries, suggest technical AI-driven "
            f"architectures (e.g., using RL for scheduling, XAI for transparency, or multimodal fusion)."
        )
        
        return {
            'system_prompt': personality_prompt + analytical_appendix,
            'cognitive_state': cognitive_state,
            'emotion_state': emotion_state,
            'user_name': user_name,
            'user_id': self.user_id
        }

    def get_service_status(self) -> Dict:
        """Get status of all services"""
        return {
            'services_initialized': self._services_initialized,
            'user_id': self.user_id,
            'services': {
                'cognitive_fusion': self.cognitive_fusion is not None,
                'emotion_detector': self.emotion_detector is not None,
                'behavior_modeler': self.behavior_modeler is not None,
                'rl_scheduler': self.rl_scheduler is not None,
                'xai_layer': self.xai_layer is not None,
                'personality_engine': self.personality_engine is not None,
                'habit_predictor': self.habit_predictor is not None,
                'anomaly_detector': self.anomaly_detector is not None,
                'analytics_dashboard': self.analytics_dashboard is not None
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def save_all_state(self):
        """Save state of all services"""
        if self.personality_engine: self.personality_engine._save_personality()
        if self.habit_predictor: self.habit_predictor._save_data()
        if self.anomaly_detector: self.anomaly_detector._save_data()
        if self.analytics_dashboard: self.analytics_dashboard._save_data()
        
        return {'saved': True, 'timestamp': datetime.now().isoformat()}


# Factory function for creating service instance
def create_ai_services(user_id: str) -> CognitiveAIServices:
    """
    Factory function to create AI services for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Initialized CognitiveAIServices instance
    """
    with _SERVICE_CACHE_LOCK:
        cached = _SERVICE_CACHE.get(user_id)
        if cached is not None:
            return cached

        service = CognitiveAIServices(user_id)
        _SERVICE_CACHE[user_id] = service
        return service


def clear_ai_services_cache(user_id: Optional[str] = None) -> None:
    """Clear cached service instances (useful for tests or manual resets)."""
    with _SERVICE_CACHE_LOCK:
        if user_id is None:
            _SERVICE_CACHE.clear()
            return
        _SERVICE_CACHE.pop(user_id, None)


# REST API-ready response formatters
class APIResponseFormatter:
    """Format responses for REST API consumption"""
    
    @staticmethod
    def success(data: Any, message: str = None) -> Dict:
        """Format successful response"""
        response = {
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        if message:
            response['message'] = message
        return response
    
    @staticmethod
    def error(error: str, code: str = None) -> Dict:
        """Format error response"""
        response = {
            'success': False,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        if code:
            response['code'] = code
        return response
    
    @staticmethod
    def paginated(data: List, page: int, per_page: int, total: int) -> Dict:
        """Format paginated response"""
        return {
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            },
            'timestamp': datetime.now().isoformat()
        }
