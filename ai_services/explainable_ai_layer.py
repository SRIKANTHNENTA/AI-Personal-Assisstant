"""
Explainable AI (XAI) Layer
Provides transparency and interpretability for AI decisions.
Uses SHAP, LIME, and custom explanation generation techniques.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path


class ExplainableAILayer:
    """
    Explainable AI Layer providing interpretable explanations for AI decisions.
    Supports multiple explanation methods including SHAP, LIME, and rule-based.
    """
    
    def __init__(self):
        self.shap_available = False
        self.lime_available = False
        
        # Initialize explanation libraries
        self._init_libraries()
        
        # Explanation templates
        self.templates = {
            'task_scheduling': {
                'positive': [
                    "Based on your {feature}, scheduling at {time} is recommended because {reason}.",
                    "Your {feature} suggests {time} would be optimal: {reason}.",
                    "Given your current {feature}, {time} aligns well because {reason}."
                ],
                'negative': [
                    "Avoid {time} because your {feature} indicates {reason}.",
                    "Your {feature} suggests {time} may not be ideal: {reason}."
                ]
            },
            'emotion_detection': {
                'confident': [
                    "Detected {emotion} with high confidence based on {features}.",
                    "Strong indicators of {emotion}: {features}."
                ],
                'uncertain': [
                    "Possible {emotion}, uncertain due to mixed signals from {features}.",
                    "Leaning towards {emotion}, but {features} show ambiguity."
                ]
            },
            'recommendation': {
                'general': [
                    "Recommending {action} because {reason}.",
                    "Based on {context}, {action} would be beneficial: {reason}."
                ]
            }
        }
        
        # Feature importance cache
        self.feature_importance_cache = {}
        
        # Decision history for audit trail
        self.decision_history = []
        self.max_history = 1000
    
    def _init_libraries(self):
        """Initialize SHAP and LIME if available"""
        try:
            import shap
            self.shap = shap
            self.shap_available = True
            print("✅ SHAP explainer available")
        except ImportError:
            print("⚠️ SHAP not installed. Install with: pip install shap")
        
        try:
            import lime
            import lime.lime_tabular
            self.lime = lime
            self.lime_tabular = lime.lime_tabular
            self.lime_available = True
            print("✅ LIME explainer available")
        except ImportError:
            print("⚠️ LIME not installed. Install with: pip install lime")
    
    def explain_decision(self,
                        decision_type: str,
                        decision: Any,
                        features: Dict,
                        model: Any = None,
                        method: str = 'auto') -> Dict:
        """
        Generate explanation for a decision.
        
        Args:
            decision_type: Type of decision ('task_scheduling', 'emotion_detection', etc.)
            decision: The actual decision/prediction made
            features: Input features used for the decision
            model: ML model (if applicable) for SHAP/LIME explanations
            method: Explanation method ('shap', 'lime', 'rule_based', 'auto')
            
        Returns:
            Dict with explanation, feature importance, and confidence
        """
        if method == 'auto':
            method = self._select_method(model)
        
        # Generate feature importance
        if method == 'shap' and model is not None and self.shap_available:
            importance = self._shap_explain(model, features)
        elif method == 'lime' and model is not None and self.lime_available:
            importance = self._lime_explain(model, features)
        else:
            importance = self._rule_based_importance(decision_type, features)
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            decision_type, decision, features, importance
        )
        
        # Calculate confidence
        confidence = self._calculate_explanation_confidence(importance)
        
        # Build result
        result = {
            'success': True,
            'decision_type': decision_type,
            'decision': decision,
            'explanation': explanation,
            'feature_importance': importance,
            'confidence': confidence,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
        
        # Record for audit
        self._record_decision(result)
        
        return result
    
    def _select_method(self, model: Any) -> str:
        """Select best available explanation method"""
        if model is not None:
            if self.shap_available:
                return 'shap'
            elif self.lime_available:
                return 'lime'
        return 'rule_based'
    
    def _shap_explain(self, model: Any, features: Dict) -> Dict:
        """Generate SHAP feature importance"""
        try:
            # Convert features to array
            feature_names = list(features.keys())
            feature_values = np.array([list(features.values())])
            
            # Create explainer
            try:
                # Try TreeExplainer for tree-based models
                explainer = self.shap.TreeExplainer(model)
            except:
                # Fallback to KernelExplainer
                explainer = self.shap.KernelExplainer(
                    model.predict if hasattr(model, 'predict') else model,
                    feature_values
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(feature_values)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Build importance dict
            importance = {}
            for i, name in enumerate(feature_names):
                importance[name] = {
                    'value': float(features[name]) if np.isscalar(features[name]) else str(features[name]),
                    'shap_value': float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i]),
                    'importance': abs(float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i])),
                    'direction': 'positive' if (shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]) > 0 else 'negative'
                }
            
            # Sort by importance
            importance = dict(sorted(
                importance.items(), 
                key=lambda x: x[1]['importance'], 
                reverse=True
            ))
            
            return importance
            
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return self._rule_based_importance('general', features)
    
    def _lime_explain(self, model: Any, features: Dict) -> Dict:
        """Generate LIME feature importance"""
        try:
            feature_names = list(features.keys())
            feature_values = np.array([list(features.values())])
            
            # Create LIME explainer
            explainer = self.lime_tabular.LimeTabularExplainer(
                feature_values,
                feature_names=feature_names,
                mode='regression' if not hasattr(model, 'predict_proba') else 'classification'
            )
            
            # Generate explanation
            if hasattr(model, 'predict_proba'):
                exp = explainer.explain_instance(
                    feature_values[0], model.predict_proba, num_features=len(feature_names)
                )
            else:
                exp = explainer.explain_instance(
                    feature_values[0], model.predict, num_features=len(feature_names)
                )
            
            # Build importance dict
            importance = {}
            for name, weight in exp.as_list():
                # Parse feature name
                original_name = name.split(' ')[0] if ' ' in name else name
                if original_name in feature_names:
                    importance[original_name] = {
                        'value': str(features.get(original_name, 'N/A')),
                        'lime_weight': float(weight),
                        'importance': abs(float(weight)),
                        'direction': 'positive' if weight > 0 else 'negative'
                    }
            
            return dict(sorted(
                importance.items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            ))
            
        except Exception as e:
            print(f"LIME explanation error: {e}")
            return self._rule_based_importance('general', features)
    
    def _rule_based_importance(self, decision_type: str, features: Dict) -> Dict:
        """Generate rule-based feature importance when ML methods unavailable"""
        # Domain knowledge-based importance weights
        importance_weights = {
            'task_scheduling': {
                'priority': 0.25,
                'complexity': 0.20,
                'deadline_hours': 0.20,
                'emotional_state': 0.15,
                'energy_level': 0.10,
                'current_hour': 0.10
            },
            'emotion_detection': {
                'text_sentiment': 0.35,
                'facial_expression': 0.35,
                'voice_tone': 0.20,
                'behavioral_context': 0.10
            },
            'recommendation': {
                'user_preference': 0.30,
                'historical_success': 0.25,
                'current_context': 0.25,
                'system_suggestion': 0.20
            }
        }
        
        weights = importance_weights.get(decision_type, {})
        
        importance = {}
        for feature, value in features.items():
            weight = weights.get(feature, 0.1)
            
            # Determine direction based on value
            if isinstance(value, (int, float)):
                direction = 'positive' if value > 0.5 else 'negative'
            elif isinstance(value, str):
                positive_terms = ['high', 'good', 'positive', 'happy', 'focused']
                direction = 'positive' if any(t in value.lower() for t in positive_terms) else 'negative'
            else:
                direction = 'neutral'
            
            importance[feature] = {
                'value': str(value),
                'importance': weight,
                'direction': direction,
                'weight': weight
            }
        
        return dict(sorted(
            importance.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        ))
    
    def _generate_explanation(self,
                             decision_type: str,
                             decision: Any,
                             features: Dict,
                             importance: Dict) -> str:
        """Generate natural language explanation"""
        # Get top important features
        top_features = list(importance.items())[:3]
        
        if decision_type == 'task_scheduling':
            return self._explain_scheduling(decision, features, top_features)
        elif decision_type == 'emotion_detection':
            return self._explain_emotion(decision, features, top_features)
        elif decision_type == 'recommendation':
            return self._explain_recommendation(decision, features, top_features)
        else:
            return self._explain_general(decision, features, top_features)
    
    def _explain_scheduling(self, decision: Any, features: Dict, 
                           top_features: List) -> str:
        """Generate scheduling explanation"""
        parts = []
        
        # Main decision
        if isinstance(decision, dict):
            time = decision.get('recommended_hour', decision.get('time', 'unknown'))
            parts.append(f"Scheduling task for {time}:00")
        else:
            parts.append(f"Scheduling decision: {decision}")
        
        # Key factors
        factors = []
        for feature, info in top_features:
            value = info['value']
            direction = info['direction']
            
            if feature == 'priority' and 'high' in str(value).lower():
                factors.append("high priority requires immediate attention")
            elif feature == 'energy_level':
                factors.append(f"your {value} energy level")
            elif feature == 'emotional_state':
                factors.append(f"your current {value} state")
            elif feature == 'complexity':
                factors.append(f"{value} task complexity")
        
        if factors:
            parts.append(f"Key factors: {', '.join(factors)}")
        
        return ". ".join(parts) + "."
    
    def _explain_emotion(self, decision: Any, features: Dict,
                        top_features: List) -> str:
        """Generate emotion detection explanation"""
        if isinstance(decision, dict):
            emotion = decision.get('emotion', decision.get('primary_emotion', 'neutral'))
            confidence = decision.get('confidence', 0.5)
        else:
            emotion = str(decision)
            confidence = 0.5
        
        parts = [f"Detected {emotion} emotion"]
        
        if confidence > 0.8:
            parts.append("with high confidence")
        elif confidence > 0.5:
            parts.append("with moderate confidence")
        else:
            parts.append("with some uncertainty")
        
        # Evidence from features
        evidence = []
        for feature, info in top_features:
            if info['importance'] > 0.1:
                if feature == 'text_sentiment':
                    evidence.append(f"text analysis showing {info['value']}")
                elif feature == 'facial_expression':
                    evidence.append(f"facial expression ({info['value']})")
                elif feature == 'voice_tone':
                    evidence.append(f"voice tone indicators")
        
        if evidence:
            parts.append(f"based on {', '.join(evidence)}")
        
        return ". ".join(parts) + "."
    
    def _explain_recommendation(self, decision: Any, features: Dict,
                               top_features: List) -> str:
        """Generate recommendation explanation"""
        if isinstance(decision, dict):
            action = decision.get('action', decision.get('recommendation', str(decision)))
        else:
            action = str(decision)
        
        parts = [f"Recommending: {action}"]
        
        reasons = []
        for feature, info in top_features:
            if info['direction'] == 'positive':
                reasons.append(f"favorable {feature.replace('_', ' ')}")
            else:
                reasons.append(f"considering {feature.replace('_', ' ')}")
        
        if reasons:
            parts.append(f"Reasoning: {'; '.join(reasons[:2])}")
        
        return " ".join(parts)
    
    def _explain_general(self, decision: Any, features: Dict,
                        top_features: List) -> str:
        """Generate general explanation"""
        factors = [f"{name}: {info['value']}" for name, info in top_features[:3]]
        return f"Decision: {decision}. Key factors: {', '.join(factors)}."
    
    def _calculate_explanation_confidence(self, importance: Dict) -> float:
        """Calculate confidence in the explanation"""
        if not importance:
            return 0.5
        
        # Higher confidence if few features dominate
        importances = [info.get('importance', 0) for info in importance.values()]
        
        if len(importances) < 2:
            return 0.5
        
        # Calculate concentration of importance
        total = sum(importances)
        if total == 0:
            return 0.5
        
        # Gini coefficient-like measure
        sorted_imp = sorted(importances, reverse=True)
        top_2_share = sum(sorted_imp[:2]) / total
        
        return min(0.95, 0.5 + top_2_share * 0.45)
    
    def _record_decision(self, result: Dict):
        """Record decision for audit trail"""
        self.decision_history.append({
            'timestamp': result['timestamp'],
            'type': result['decision_type'],
            'summary': result['explanation'][:200]
        })
        
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
    
    def get_decision_audit(self, n: int = 10, decision_type: str = None) -> List[Dict]:
        """Get recent decision audit trail"""
        history = self.decision_history
        
        if decision_type:
            history = [d for d in history if d['type'] == decision_type]
        
        return history[-n:]


class TaskRecommendationExplainer:
    """
    Specialized explainer for task scheduling recommendations.
    Provides detailed justification for scheduling decisions.
    """
    
    def __init__(self, xai_layer: Optional[ExplainableAILayer] = None):
        self.xai = xai_layer or ExplainableAILayer()
        
        # Domain knowledge for explanations
        self.time_of_day_context = {
            range(5, 9): "early morning (fresh focus)",
            range(9, 12): "mid-morning (peak productivity)",
            range(12, 14): "midday (post-lunch dip)",
            range(14, 17): "afternoon (sustained work)",
            range(17, 20): "early evening (winding down)",
            range(20, 24): "evening (low energy)"
        }
        
        self.complexity_energy_match = {
            ('complex', 'high'): "Your high energy is ideal for complex work",
            ('complex', 'medium'): "Complex task may be challenging with current energy",
            ('complex', 'low'): "Consider a simpler task when energy is low",
            ('simple', 'high'): "High energy could tackle more challenging work",
            ('simple', 'medium'): "Good match for moderate difficulty",
            ('simple', 'low'): "Simple task appropriate for current energy"
        }
    
    def explain_schedule(self,
                        task: Dict,
                        recommended_time: int,
                        user_state: Dict,
                        alternatives: List[Dict] = None) -> Dict:
        """
        Generate detailed explanation for a scheduling recommendation.
        
        Args:
            task: Task details
            recommended_time: Recommended hour (0-23)
            user_state: User's current state
            alternatives: Alternative time slots considered
            
        Returns:
            Comprehensive explanation
        """
        # Build features dict
        features = {
            'priority': task.get('priority', 'medium'),
            'complexity': task.get('complexity', 'medium'),
            'estimated_duration': task.get('estimated_duration', 30),
            'energy_level': user_state.get('energy', 'medium'),
            'emotional_state': user_state.get('emotion', 'neutral'),
            'current_hour': user_state.get('current_hour', datetime.now().hour)
        }
        
        # Time context
        time_context = "unknown time"
        for time_range, context in self.time_of_day_context.items():
            if recommended_time in time_range:
                time_context = context
                break
        
        # Energy-complexity match
        complexity = task.get('complexity', 'medium').lower()
        energy = user_state.get('energy', 'medium').lower()
        energy_match = self.complexity_energy_match.get(
            (complexity, energy),
            "Reasonable match for current conditions"
        )
        
        # Build explanation components
        main_reasoning = []
        
        # Priority reasoning
        priority = task.get('priority', 'medium').lower()
        if priority == 'high':
            main_reasoning.append("High priority task scheduled at earliest optimal slot")
        elif priority == 'low':
            main_reasoning.append("Low priority allows flexible scheduling")
        
        # Time reasoning
        main_reasoning.append(f"Scheduled during {time_context}")
        
        # Energy reasoning
        main_reasoning.append(energy_match)
        
        # Deadline reasoning
        deadline = task.get('deadline_hours')
        if deadline and deadline < 4:
            main_reasoning.append("Urgent: deadline approaching")
        
        # Alternative comparison
        comparison = None
        if alternatives and len(alternatives) > 1:
            second_best = alternatives[1] if alternatives[0]['hour'] == recommended_time else alternatives[0]
            comparison = f"Alternative: {second_best['hour']:02d}:00 (score: {second_best.get('score', 0):.2f})"
        
        # Emotional consideration
        emotion = user_state.get('emotion', 'neutral').lower()
        emotional_advice = None
        if emotion in ['stressed', 'anxious']:
            emotional_advice = "Consider a short break before starting if feeling stressed"
        elif emotion in ['tired', 'sad']:
            emotional_advice = "Task scheduled with buffer for current energy level"
        
        return {
            'recommended_time': recommended_time,
            'time_range': f"{recommended_time:02d}:00 - {(recommended_time + (task.get('estimated_duration') or 30) // 60 + 1) % 24:02d}:00",
            'time_context': time_context,
            'main_reasoning': main_reasoning,
            'energy_match': energy_match,
            'alternative_comparison': comparison,
            'emotional_advice': emotional_advice,
            'feature_analysis': features,
            'confidence_factors': {
                'priority_weight': 0.25 if priority == 'high' else (0.15 if priority == 'medium' else 0.1),
                'energy_alignment': 0.9 if (complexity == 'complex' and energy == 'high') or (complexity == 'simple' and energy != 'high') else 0.6,
                'time_suitability': 0.85 if 9 <= recommended_time <= 11 else (0.7 if 14 <= recommended_time <= 16 else 0.5)
            },
            'summary': f"Scheduled for {recommended_time:02d}:00 ({time_context}). {main_reasoning[0]}. {energy_match}."
        }


class EmotionExplainer:
    """
    Specialized explainer for emotion detection results.
    Provides transparency into multimodal emotion analysis.
    """
    
    def __init__(self, xai_layer: Optional[ExplainableAILayer] = None):
        self.xai = xai_layer or ExplainableAILayer()
        
        # Emotion indicators
        self.text_indicators = {
            'happy': ['joy', 'excited', 'great', 'wonderful', 'love', 'amazing'],
            'sad': ['sad', 'down', 'depressed', 'unhappy', 'miserable', 'cry'],
            'angry': ['angry', 'frustrated', 'mad', 'furious', 'annoyed'],
            'anxious': ['worried', 'nervous', 'anxious', 'stressed', 'concerned'],
            'neutral': ['okay', 'fine', 'alright', 'normal']
        }
        
        self.facial_cues = {
            'happy': 'raised cheeks, smile',
            'sad': 'lowered eyes, downturned mouth',
            'angry': 'furrowed brow, tense jaw',
            'surprised': 'raised eyebrows, open mouth',
            'fearful': 'wide eyes, tense expression',
            'neutral': 'relaxed features'
        }
        
        self.voice_cues = {
            'happy': 'higher pitch, varied intonation',
            'sad': 'lower pitch, monotone',
            'angry': 'increased volume, sharp tones',
            'anxious': 'faster speech, higher pitch'
        }
    
    def explain_emotion(self,
                       detected_emotion: str,
                       modality_results: Dict,
                       confidence: float) -> Dict:
        """
        Explain emotion detection result.
        
        Args:
            detected_emotion: The detected emotion
            modality_results: Results from each modality
            confidence: Overall confidence score
            
        Returns:
            Detailed explanation of emotion detection
        """
        evidence = []
        modality_contributions = []
        
        # Text analysis evidence
        text_result = modality_results.get('text', {})
        if text_result.get('success'):
            text_emotion = text_result.get('emotion', 'neutral')
            text_conf = text_result.get('confidence', 0)
            evidence.append(f"Text analysis: {text_emotion} ({text_conf:.0%} confident)")
            modality_contributions.append({
                'modality': 'text',
                'emotion': text_emotion,
                'confidence': text_conf,
                'indicators': self._find_text_indicators(text_result)
            })
        
        # Voice analysis evidence
        voice_result = modality_results.get('voice', {})
        if voice_result.get('success'):
            voice_emotion = voice_result.get('emotion', 'neutral')
            voice_conf = voice_result.get('confidence', 0)
            evidence.append(f"Voice analysis: {voice_emotion} ({voice_conf:.0%} confident)")
            modality_contributions.append({
                'modality': 'voice',
                'emotion': voice_emotion,
                'confidence': voice_conf,
                'cues': self.voice_cues.get(voice_emotion, 'standard prosody')
            })
        
        # Facial analysis evidence
        facial_result = modality_results.get('facial', {})
        if facial_result.get('success'):
            facial_emotion = facial_result.get('emotion', 'neutral')
            facial_conf = facial_result.get('confidence', 0)
            evidence.append(f"Facial analysis: {facial_emotion} ({facial_conf:.0%} confident)")
            modality_contributions.append({
                'modality': 'facial',
                'emotion': facial_emotion,
                'confidence': facial_conf,
                'cues': self.facial_cues.get(facial_emotion, 'mixed expression')
            })
        
        # Agreement analysis
        emotions_detected = [c['emotion'] for c in modality_contributions]
        agreement = emotions_detected.count(detected_emotion) / len(emotions_detected) if emotions_detected else 0
        
        # Confidence interpretation
        if confidence > 0.8:
            confidence_interpretation = "High confidence - strong agreement across modalities"
        elif confidence > 0.5:
            confidence_interpretation = "Moderate confidence - some variation in signals"
        else:
            confidence_interpretation = "Lower confidence - mixed or weak signals"
        
        # Possible alternative emotions
        all_emotions = [c['emotion'] for c in modality_contributions]
        alternatives = list(set(all_emotions) - {detected_emotion})
        
        return {
            'detected_emotion': detected_emotion,
            'confidence': confidence,
            'evidence': evidence,
            'modality_contributions': modality_contributions,
            'agreement_score': agreement,
            'confidence_interpretation': confidence_interpretation,
            'alternative_emotions': alternatives,
            'summary': f"Detected {detected_emotion} with {confidence:.0%} confidence. {confidence_interpretation}.",
            'detailed_explanation': self._build_detailed_explanation(
                detected_emotion, modality_contributions, confidence
            )
        }
    
    def _find_text_indicators(self, text_result: Dict) -> List[str]:
        """Find emotion indicators in text"""
        # This would analyze the actual text in production
        emotion = text_result.get('emotion', 'neutral')
        return self.text_indicators.get(emotion, ['no specific indicators'])[:3]
    
    def _build_detailed_explanation(self,
                                   emotion: str,
                                   contributions: List[Dict],
                                   confidence: float) -> str:
        """Build detailed multi-sentence explanation"""
        parts = [f"The system detected {emotion} as the primary emotional state."]
        
        for contrib in contributions:
            modality = contrib['modality']
            mod_emotion = contrib['emotion']
            mod_conf = contrib['confidence']
            
            if modality == 'text':
                parts.append(f"Text analysis indicated {mod_emotion} based on language patterns.")
            elif modality == 'voice':
                cues = contrib.get('cues', 'vocal patterns')
                parts.append(f"Voice analysis detected {mod_emotion} through {cues}.")
            elif modality == 'facial':
                cues = contrib.get('cues', 'facial features')
                parts.append(f"Facial analysis identified {mod_emotion} from {cues}.")
        
        if confidence > 0.7:
            parts.append("The modalities show strong agreement, supporting high confidence.")
        else:
            parts.append("Some variation between modalities suggests moderate confidence.")
        
        return " ".join(parts)


class FeatureAttributionVisualizer:
    """
    Generates visualizations of feature attributions for explanations.
    Creates data structures for frontend rendering.
    """
    
    def __init__(self):
        pass
    
    def prepare_waterfall_data(self, importance: Dict, 
                               base_value: float = 0.5) -> Dict:
        """
        Prepare data for waterfall plot visualization.
        
        Args:
            importance: Feature importance from ExplainableAILayer
            base_value: Base prediction value
            
        Returns:
            Data structure for waterfall visualization
        """
        if not importance:
            return {'success': False, 'error': 'No importance data'}
        
        cumulative = base_value
        data_points = []
        
        for feature, info in importance.items():
            imp_value = info.get('shap_value', info.get('lime_weight', info.get('importance', 0)))
            direction = info.get('direction', 'positive')
            
            # Scale importance for visualization
            if direction == 'negative':
                imp_value = -abs(imp_value)
            else:
                imp_value = abs(imp_value)
            
            data_points.append({
                'feature': feature,
                'value': info.get('value', 'N/A'),
                'contribution': imp_value,
                'start': cumulative,
                'end': cumulative + imp_value,
                'direction': 'up' if imp_value >= 0 else 'down'
            })
            
            cumulative += imp_value
        
        return {
            'success': True,
            'base_value': base_value,
            'final_value': cumulative,
            'data_points': data_points
        }
    
    def prepare_bar_chart_data(self, importance: Dict) -> Dict:
        """
        Prepare data for horizontal bar chart of feature importance.
        
        Args:
            importance: Feature importance from ExplainableAILayer
            
        Returns:
            Data structure for bar chart visualization
        """
        if not importance:
            return {'success': False, 'error': 'No importance data'}
        
        bars = []
        for feature, info in importance.items():
            imp_value = info.get('importance', 0)
            direction = info.get('direction', 'positive')
            
            bars.append({
                'feature': feature.replace('_', ' ').title(),
                'importance': imp_value,
                'value': info.get('value', 'N/A'),
                'color': '#4CAF50' if direction == 'positive' else '#F44336'
            })
        
        # Sort by importance
        bars.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return {
            'success': True,
            'bars': bars,
            'max_importance': max(abs(b['importance']) for b in bars) if bars else 1.0
        }
    
    def prepare_radar_data(self, features: Dict, 
                          categories: List[str] = None) -> Dict:
        """
        Prepare data for radar/spider chart.
        
        Args:
            features: Feature values (normalized 0-1)
            categories: Category names for each axis
            
        Returns:
            Data structure for radar visualization
        """
        if not features:
            return {'success': False, 'error': 'No feature data'}
        
        if categories is None:
            categories = list(features.keys())
        
        values = []
        for cat in categories:
            if cat in features:
                val = features[cat]
                if isinstance(val, (int, float)):
                    values.append(min(1.0, max(0.0, float(val))))
                else:
                    values.append(0.5)  # Default for non-numeric
            else:
                values.append(0.0)
        
        return {
            'success': True,
            'categories': [c.replace('_', ' ').title() for c in categories],
            'values': values
        }
