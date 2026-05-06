"""
Anomaly Detection System
Identifies unusual emotional or behavioral deviations in user patterns.
Uses statistical and ML-based methods for real-time anomaly detection.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnomalyDetectionSystem:
    """
    Detects anomalies in user emotional states, behavior patterns, and interactions.
    Uses multiple detection methods for robust anomaly identification.
    """
    
    # Anomaly types
    ANOMALY_TYPES = {
        'emotional': ['sudden_mood_shift', 'prolonged_negative', 'emotional_volatility'],
        'behavioral': ['activity_spike', 'activity_drop', 'time_anomaly', 'pattern_break'],
        'interaction': ['communication_drop', 'response_change', 'engagement_anomaly']
    }
    
    # Severity levels
    SEVERITY_LEVELS = {
        'low': {'threshold': 0.6, 'action': 'monitor'},
        'medium': {'threshold': 0.75, 'action': 'alert'},
        'high': {'threshold': 0.9, 'action': 'intervention'}
    }
    
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        
        # Data buffers (rolling windows)
        self.emotion_buffer = deque(maxlen=1000)
        self.behavior_buffer = deque(maxlen=1000)
        self.interaction_buffer = deque(maxlen=1000)
        
        # Baseline statistics
        self.baselines = {
            'emotion': {},
            'behavior': {},
            'interaction': {}
        }
        
        # Detected anomalies history
        self.anomaly_history = []
        
        # ML models
        self.isolation_forest = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.models_trained = False
        
        # Detection sensitivity
        self.sensitivity = 0.5  # 0 = less sensitive, 1 = more sensitive
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Model persistence
        self.model_dir = Path(__file__).parent / 'anomaly_models'
        self.model_dir.mkdir(exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load saved anomaly data and baselines"""
        if self.user_id:
            path = self.model_dir / f'anomaly_data_{self.user_id}.json'
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        self.baselines = data.get('baselines', self.baselines)
                        self.anomaly_history = data.get('anomaly_history', [])[-500:]
                        self.sensitivity = data.get('sensitivity', 0.5)
                except Exception as e:
                    print(f"Warning: Could not load anomaly data: {e}")
    
    def _save_data(self):
        """Save anomaly data"""
        if self.user_id:
            path = self.model_dir / f'anomaly_data_{self.user_id}.json'
            try:
                data = {
                    'baselines': self.baselines,
                    'anomaly_history': self.anomaly_history[-500:],
                    'sensitivity': self.sensitivity,
                    'last_updated': datetime.now().isoformat()
                }
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save anomaly data: {e}")
    
    def set_sensitivity(self, sensitivity: float):
        """Set detection sensitivity (0-1)"""
        self.sensitivity = np.clip(sensitivity, 0, 1)
        self._save_data()
    
    def register_alert_callback(self, callback):
        """Register callback for anomaly alerts"""
        self.alert_callbacks.append(callback)
    
    def record_emotion_state(self,
                            emotion: str,
                            valence: float,
                            arousal: float,
                            confidence: float = 0.8,
                            source: str = 'multimodal') -> Dict:
        """
        Record emotional state and check for anomalies.
        
        Args:
            emotion: Detected emotion label
            valence: Emotional valence (-1 to 1)
            arousal: Emotional arousal (0 to 1)
            confidence: Detection confidence
            source: Detection source (text, voice, facial, multimodal)
            
        Returns:
            Anomaly detection results
        """
        now = datetime.now()
        
        entry = {
            'timestamp': now.isoformat(),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'emotion': emotion,
            'valence': valence,
            'arousal': arousal,
            'confidence': confidence,
            'source': source
        }
        
        self.emotion_buffer.append(entry)
        
        # Check for emotional anomalies
        anomalies = self._detect_emotional_anomalies(entry)
        
        # Update baselines
        self._update_emotion_baseline()
        
        # Save periodically
        if len(self.emotion_buffer) % 50 == 0:
            self._save_data()
        
        return {
            'recorded': True,
            'anomalies_detected': len(anomalies) > 0,
            'anomalies': anomalies
        }
    
    def record_behavior_event(self,
                             event_type: str,
                             category: str = 'task',
                             duration: float = None,
                             metadata: Dict = None) -> Dict:
        """
        Record behavioral event and check for anomalies.
        
        Args:
            event_type: Type of event (task_start, task_complete, break, etc.)
            category: Event category (task, communication, leisure, etc.)
            duration: Duration in minutes
            metadata: Additional event metadata
            
        Returns:
            Anomaly detection results
        """
        now = datetime.now()
        
        entry = {
            'timestamp': now.isoformat(),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'event_type': event_type,
            'category': category,
            'duration': duration,
            'metadata': metadata or {}
        }
        
        self.behavior_buffer.append(entry)
        
        # Check for behavioral anomalies
        anomalies = self._detect_behavioral_anomalies(entry)
        
        # Update baselines
        self._update_behavior_baseline()
        
        return {
            'recorded': True,
            'anomalies_detected': len(anomalies) > 0,
            'anomalies': anomalies
        }
    
    def record_interaction(self,
                          message_length: int,
                          response_time: float,
                          sentiment: float,
                          intent: str = None) -> Dict:
        """
        Record interaction data and check for anomalies.
        
        Args:
            message_length: Length of user message
            response_time: Time to respond (seconds)
            sentiment: Sentiment score (-1 to 1)
            intent: Detected intent
            
        Returns:
            Anomaly detection results
        """
        now = datetime.now()
        
        entry = {
            'timestamp': now.isoformat(),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'message_length': message_length,
            'response_time': response_time,
            'sentiment': sentiment,
            'intent': intent
        }
        
        self.interaction_buffer.append(entry)
        
        # Check for interaction anomalies
        anomalies = self._detect_interaction_anomalies(entry)
        
        return {
            'recorded': True,
            'anomalies_detected': len(anomalies) > 0,
            'anomalies': anomalies
        }
    
    def _detect_emotional_anomalies(self, current: Dict) -> List[Dict]:
        """Detect emotional anomalies"""
        anomalies = []
        
        if len(self.emotion_buffer) < 10:
            return anomalies  # Need baseline data
        
        recent = list(self.emotion_buffer)[-20:]
        baseline = self.baselines.get('emotion', {})
        
        # 1. Sudden mood shift detection
        if len(recent) >= 2:
            previous = recent[-2]
            valence_shift = abs(current['valence'] - previous['valence'])
            arousal_shift = abs(current['arousal'] - previous['arousal'])
            
            threshold = 0.6 * (1 - self.sensitivity * 0.3)
            
            if valence_shift > threshold or arousal_shift > threshold:
                anomalies.append(self._create_anomaly(
                    anomaly_type='sudden_mood_shift',
                    category='emotional',
                    severity=self._calculate_severity(valence_shift),
                    details={
                        'previous_emotion': previous['emotion'],
                        'current_emotion': current['emotion'],
                        'valence_shift': valence_shift,
                        'arousal_shift': arousal_shift
                    },
                    message=f"Sudden mood shift detected: {previous['emotion']} → {current['emotion']}"
                ))
        
        # 2. Prolonged negative emotion detection
        if baseline.get('mean_valence') is not None:
            negative_threshold = -0.3
            negative_window = [e for e in recent[-10:] if e['valence'] < negative_threshold]
            
            if len(negative_window) >= 7:
                anomalies.append(self._create_anomaly(
                    anomaly_type='prolonged_negative',
                    category='emotional',
                    severity='medium',
                    details={
                        'negative_count': len(negative_window),
                        'avg_valence': np.mean([e['valence'] for e in negative_window]),
                        'emotions': [e['emotion'] for e in negative_window]
                    },
                    message='Prolonged negative emotional state detected'
                ))
        
        # 3. Emotional volatility detection
        if len(recent) >= 10:
            valences = [e['valence'] for e in recent[-10:]]
            volatility = np.std(valences)
            
            if baseline.get('valence_volatility'):
                normal_volatility = baseline['valence_volatility']
                if volatility > normal_volatility * 2:
                    anomalies.append(self._create_anomaly(
                        anomaly_type='emotional_volatility',
                        category='emotional',
                        severity=self._calculate_severity(volatility / normal_volatility / 3),
                        details={
                            'current_volatility': volatility,
                            'baseline_volatility': normal_volatility,
                            'ratio': volatility / normal_volatility
                        },
                        message='Unusual emotional volatility detected'
                    ))
        
        # 4. ML-based anomaly detection
        if SKLEARN_AVAILABLE and self.models_trained:
            ml_anomaly = self._ml_anomaly_check([
                current['valence'],
                current['arousal'],
                current['hour'],
                current['day_of_week']
            ])
            if ml_anomaly['is_anomaly']:
                anomalies.append(self._create_anomaly(
                    anomaly_type='ml_detected_emotional',
                    category='emotional',
                    severity=self._calculate_severity(ml_anomaly['score']),
                    details={'anomaly_score': ml_anomaly['score']},
                    message='ML model detected unusual emotional pattern'
                ))
        
        # Record and alert
        for anomaly in anomalies:
            self._record_anomaly(anomaly)
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, current: Dict) -> List[Dict]:
        """Detect behavioral anomalies"""
        anomalies = []
        
        if len(self.behavior_buffer) < 20:
            return anomalies
        
        recent = list(self.behavior_buffer)
        baseline = self.baselines.get('behavior', {})
        
        # 1. Activity spike/drop detection
        today = datetime.now().date()
        today_events = [e for e in recent if e['timestamp'][:10] == str(today)]
        
        hour = current['hour']
        hourly_baseline = baseline.get(f'hourly_activity_{hour}', {})
        
        if hourly_baseline.get('mean'):
            current_hour_count = len([e for e in today_events if e['hour'] == hour])
            mean = hourly_baseline['mean']
            std = hourly_baseline.get('std', mean * 0.3)
            
            z_score = (current_hour_count - mean) / max(std, 0.1)
            
            threshold = 2 * (1 - self.sensitivity * 0.3)
            
            if z_score > threshold:
                anomalies.append(self._create_anomaly(
                    anomaly_type='activity_spike',
                    category='behavioral',
                    severity='medium',
                    details={
                        'hour': hour,
                        'count': current_hour_count,
                        'baseline_mean': mean,
                        'z_score': z_score
                    },
                    message=f'Unusual activity spike at hour {hour}'
                ))
            elif z_score < -threshold:
                anomalies.append(self._create_anomaly(
                    anomaly_type='activity_drop',
                    category='behavioral',
                    severity='low',
                    details={
                        'hour': hour,
                        'count': current_hour_count,
                        'baseline_mean': mean,
                        'z_score': z_score
                    },
                    message=f'Unusual activity drop at hour {hour}'
                ))
        
        # 2. Time anomaly (activity at unusual times)
        active_hours = baseline.get('active_hours', list(range(8, 22)))
        if hour not in active_hours and len(active_hours) > 5:
            anomalies.append(self._create_anomaly(
                anomaly_type='time_anomaly',
                category='behavioral',
                severity='low',
                details={
                    'hour': hour,
                    'usual_active_hours': active_hours[:5]  # Show first few
                },
                message=f'Activity at unusual hour ({hour}:00)'
            ))
        
        # 3. Pattern break detection
        day = current['day_of_week']
        day_pattern = baseline.get(f'day_pattern_{day}', {})
        
        if day_pattern.get('typical_categories'):
            typical = day_pattern['typical_categories']
            if current['category'] not in typical and len(typical) >= 3:
                anomalies.append(self._create_anomaly(
                    anomaly_type='pattern_break',
                    category='behavioral',
                    severity='low',
                    details={
                        'current_category': current['category'],
                        'typical_categories': typical
                    },
                    message=f"Unusual activity category '{current['category']}'"
                ))
        
        for anomaly in anomalies:
            self._record_anomaly(anomaly)
        
        return anomalies
    
    def _detect_interaction_anomalies(self, current: Dict) -> List[Dict]:
        """Detect interaction anomalies"""
        anomalies = []
        
        if len(self.interaction_buffer) < 20:
            return anomalies
        
        recent = list(self.interaction_buffer)[-50:]
        baseline = self.baselines.get('interaction', {})
        
        # 1. Communication pattern change
        if baseline.get('mean_message_length'):
            mean_length = baseline['mean_message_length']
            std_length = baseline.get('std_message_length', mean_length * 0.5)
            
            z_score = (current['message_length'] - mean_length) / max(std_length, 1)
            
            if abs(z_score) > 2.5:
                direction = 'much longer' if z_score > 0 else 'much shorter'
                anomalies.append(self._create_anomaly(
                    anomaly_type='communication_change',
                    category='interaction',
                    severity='low',
                    details={
                        'message_length': current['message_length'],
                        'baseline_mean': mean_length,
                        'z_score': z_score,
                        'direction': direction
                    },
                    message=f'Messages are {direction} than usual'
                ))
        
        # 2. Response time anomaly
        if baseline.get('mean_response_time') and current['response_time']:
            mean_rt = baseline['mean_response_time']
            std_rt = baseline.get('std_response_time', mean_rt * 0.5)
            
            z_score = (current['response_time'] - mean_rt) / max(std_rt, 0.1)
            
            if z_score > 3:  # Much slower responses
                anomalies.append(self._create_anomaly(
                    anomaly_type='response_change',
                    category='interaction',
                    severity='medium',
                    details={
                        'response_time': current['response_time'],
                        'baseline_mean': mean_rt,
                        'z_score': z_score
                    },
                    message='Response time significantly slower than usual'
                ))
        
        # 3. Sentiment shift detection
        if baseline.get('mean_sentiment'):
            recent_sentiments = [e['sentiment'] for e in recent[-10:]]
            current_avg = np.mean(recent_sentiments)
            baseline_sentiment = baseline['mean_sentiment']
            
            shift = current_avg - baseline_sentiment
            threshold = 0.4 * (1 - self.sensitivity * 0.2)
            
            if abs(shift) > threshold:
                direction = 'more positive' if shift > 0 else 'more negative'
                anomalies.append(self._create_anomaly(
                    anomaly_type='engagement_anomaly',
                    category='interaction',
                    severity='medium' if shift < 0 else 'low',
                    details={
                        'recent_avg_sentiment': current_avg,
                        'baseline_sentiment': baseline_sentiment,
                        'shift': shift
                    },
                    message=f'Recent interactions are {direction} than usual'
                ))
        
        for anomaly in anomalies:
            self._record_anomaly(anomaly)
        
        return anomalies
    
    def _create_anomaly(self,
                       anomaly_type: str,
                       category: str,
                       severity: str,
                       details: Dict,
                       message: str) -> Dict:
        """Create anomaly record"""
        return {
            'id': f"anomaly_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'type': anomaly_type,
            'category': category,
            'severity': severity,
            'details': details,
            'message': message,
            'resolved': False
        }
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity level from score"""
        if score >= 0.9:
            return 'high'
        elif score >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _record_anomaly(self, anomaly: Dict):
        """Record anomaly and trigger alerts"""
        self.anomaly_history.append(anomaly)
        
        # Trigger callbacks for medium/high severity
        severity = anomaly.get('severity', 'low')
        if severity in ['medium', 'high']:
            for callback in self.alert_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    print(f"Alert callback error: {e}")
    
    def _update_emotion_baseline(self):
        """Update emotion baselines from buffer"""
        if len(self.emotion_buffer) < 50:
            return
        
        recent = list(self.emotion_buffer)[-100:]
        
        valences = [e['valence'] for e in recent]
        arousals = [e['arousal'] for e in recent]
        
        self.baselines['emotion'] = {
            'mean_valence': np.mean(valences),
            'std_valence': np.std(valences),
            'mean_arousal': np.mean(arousals),
            'std_arousal': np.std(arousals),
            'valence_volatility': np.std(valences),
            'emotion_distribution': self._calculate_distribution([e['emotion'] for e in recent]),
            'updated_at': datetime.now().isoformat()
        }
    
    def _update_behavior_baseline(self):
        """Update behavior baselines from buffer"""
        if len(self.behavior_buffer) < 50:
            return
        
        recent = list(self.behavior_buffer)[-200:]
        
        # Hourly activity patterns
        for hour in range(24):
            hour_events = [e for e in recent if e['hour'] == hour]
            if hour_events:
                # Group by date and count
                dates = set(e['timestamp'][:10] for e in hour_events)
                counts = [len([e for e in hour_events if e['timestamp'][:10] == d]) for d in dates]
                self.baselines['behavior'][f'hourly_activity_{hour}'] = {
                    'mean': np.mean(counts) if counts else 0,
                    'std': np.std(counts) if len(counts) > 1 else 0
                }
        
        # Active hours
        hour_counts = {}
        for e in recent:
            hour_counts[e['hour']] = hour_counts.get(e['hour'], 0) + 1
        
        self.baselines['behavior']['active_hours'] = [
            h for h, c in sorted(hour_counts.items(), key=lambda x: -x[1])[:10]
        ]
        
        # Day patterns
        for day in range(7):
            day_events = [e for e in recent if e['day_of_week'] == day]
            if day_events:
                categories = [e['category'] for e in day_events]
                category_counts = {}
                for c in categories:
                    category_counts[c] = category_counts.get(c, 0) + 1
                typical = [c for c, _ in sorted(category_counts.items(), key=lambda x: -x[1])[:5]]
                self.baselines['behavior'][f'day_pattern_{day}'] = {
                    'typical_categories': typical
                }
    
    def _calculate_distribution(self, items: List) -> Dict:
        """Calculate item distribution"""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        total = len(items)
        return {k: v / total for k, v in counts.items()}
    
    def _ml_anomaly_check(self, features: List) -> Dict:
        """Check for anomalies using ML model"""
        if not self.isolation_forest:
            return {'is_anomaly': False, 'score': 0}
        
        try:
            scaled = self.scaler.transform([features])
            score = -self.isolation_forest.decision_function(scaled)[0]
            prediction = self.isolation_forest.predict(scaled)[0]
            
            return {
                'is_anomaly': prediction == -1,
                'score': float(score)
            }
        except Exception:
            return {'is_anomaly': False, 'score': 0}
    
    def train_anomaly_models(self) -> Dict:
        """Train ML anomaly detection models"""
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'scikit-learn not available'}
        
        # Collect training data from emotion buffer
        if len(self.emotion_buffer) < 50:
            return {'success': False, 'error': 'Insufficient data for training'}
        
        X = []
        for entry in self.emotion_buffer:
            X.append([
                entry['valence'],
                entry['arousal'],
                entry['hour'],
                entry['day_of_week']
            ])
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            self.isolation_forest = IsolationForest(
                contamination=0.05 + self.sensitivity * 0.1,
                random_state=42
            )
            self.isolation_forest.fit(X_scaled)
            
            self.models_trained = True
            
            return {
                'success': True,
                'training_samples': len(X),
                'contamination': 0.05 + self.sensitivity * 0.1
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_anomaly_summary(self, 
                           days: int = 7,
                           category: str = None) -> Dict:
        """
        Get summary of anomalies over specified period.
        
        Args:
            days: Number of days to look back
            category: Filter by category (emotional, behavioral, interaction)
            
        Returns:
            Anomaly summary statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_anomalies = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]
        
        if category:
            recent_anomalies = [a for a in recent_anomalies if a['category'] == category]
        
        if not recent_anomalies:
            return {
                'period_days': days,
                'total_anomalies': 0,
                'category_filter': category,
                'anomalies': []
            }
        
        # Group by type
        by_type = {}
        for a in recent_anomalies:
            t = a['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(a)
        
        # Group by severity
        by_severity = {'low': 0, 'medium': 0, 'high': 0}
        for a in recent_anomalies:
            by_severity[a['severity']] = by_severity.get(a['severity'], 0) + 1
        
        # Time distribution
        by_hour = {}
        for a in recent_anomalies:
            hour = datetime.fromisoformat(a['timestamp']).hour
            by_hour[hour] = by_hour.get(hour, 0) + 1
        
        return {
            'period_days': days,
            'total_anomalies': len(recent_anomalies),
            'category_filter': category,
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_severity': by_severity,
            'by_hour': by_hour,
            'recent_anomalies': recent_anomalies[-10:]  # Last 10
        }
    
    def get_current_risk_assessment(self) -> Dict:
        """
        Get current risk assessment based on recent anomalies.
        
        Returns:
            Risk assessment with recommendations
        """
        # Check last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]
        
        # Calculate risk score
        risk_score = 0
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.5}
        
        for anomaly in recent:
            weight = severity_weights.get(anomaly['severity'], 0.1)
            # Decay based on time
            time_ago = (datetime.now() - datetime.fromisoformat(anomaly['timestamp'])).seconds / 3600
            decay = np.exp(-time_ago / 12)  # Half-life of 12 hours
            risk_score += weight * decay
        
        risk_score = min(1.0, risk_score)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'high'
            recommendation = 'Consider immediate intervention or check-in'
        elif risk_score >= 0.4:
            risk_level = 'moderate'
            recommendation = 'Monitor closely and consider proactive engagement'
        elif risk_score >= 0.2:
            risk_level = 'low'
            recommendation = 'Continue normal monitoring'
        else:
            risk_level = 'minimal'
            recommendation = 'No action needed'
        
        # Identify primary concerns
        concerns = []
        high_severity = [a for a in recent if a['severity'] == 'high']
        if high_severity:
            concerns.extend([a['message'] for a in high_severity[-3:]])
        
        emotional_anomalies = [a for a in recent if a['category'] == 'emotional']
        if len(emotional_anomalies) >= 3:
            concerns.append('Multiple emotional anomalies detected')
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'recent_anomaly_count': len(recent),
            'primary_concerns': concerns,
            'assessment_time': datetime.now().isoformat()
        }
    
    def resolve_anomaly(self, anomaly_id: str, resolution_notes: str = None) -> bool:
        """Mark an anomaly as resolved"""
        for anomaly in self.anomaly_history:
            if anomaly.get('id') == anomaly_id:
                anomaly['resolved'] = True
                anomaly['resolution_notes'] = resolution_notes
                anomaly['resolved_at'] = datetime.now().isoformat()
                self._save_data()
                return True
        return False
    
    def get_wellness_insights(self) -> Dict:
        """
        Generate wellness insights based on anomaly patterns.
        
        Returns:
            Insights and recommendations for user wellness
        """
        # Analyze patterns over last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        recent = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]
        
        insights = []
        
        # Check emotional patterns
        emotional = [a for a in recent if a['category'] == 'emotional']
        if emotional:
            mood_shifts = len([a for a in emotional if a['type'] == 'sudden_mood_shift'])
            prolonged_negative = len([a for a in emotional if a['type'] == 'prolonged_negative'])
            
            if prolonged_negative >= 3:
                insights.append({
                    'category': 'emotional',
                    'finding': 'Pattern of prolonged negative emotions detected',
                    'suggestion': 'Consider stress management techniques or speaking with a professional',
                    'severity': 'medium'
                })
            
            if mood_shifts >= 5:
                insights.append({
                    'category': 'emotional',
                    'finding': 'Frequent mood fluctuations observed',
                    'suggestion': 'Regular sleep and exercise may help stabilize mood',
                    'severity': 'low'
                })
        
        # Check behavioral patterns
        behavioral = [a for a in recent if a['category'] == 'behavioral']
        if behavioral:
            time_anomalies = [a for a in behavioral if a['type'] == 'time_anomaly']
            if len(time_anomalies) >= 5:
                insights.append({
                    'category': 'behavioral',
                    'finding': 'Irregular activity schedule detected',
                    'suggestion': 'Establishing a consistent routine may improve wellbeing',
                    'severity': 'low'
                })
        
        # Check interaction patterns
        interaction = [a for a in recent if a['category'] == 'interaction']
        if interaction:
            negative_engagement = [a for a in interaction if 'negative' in a.get('message', '').lower()]
            if len(negative_engagement) >= 3:
                insights.append({
                    'category': 'interaction',
                    'finding': 'Communication sentiment trending negative',
                    'suggestion': 'Consider what might be affecting your mood recently',
                    'severity': 'medium'
                })
        
        # Overall assessment
        total_anomalies = len(recent)
        days_analyzed = 30
        anomaly_rate = total_anomalies / days_analyzed
        
        if anomaly_rate < 0.5:
            overall = 'excellent'
            message = 'Your patterns appear stable and healthy'
        elif anomaly_rate < 1.5:
            overall = 'good'
            message = 'Minor deviations detected but overall patterns are healthy'
        elif anomaly_rate < 3:
            overall = 'moderate'
            message = 'Some patterns warrant attention'
        else:
            overall = 'concerning'
            message = 'Multiple areas show significant deviations from baseline'
        
        return {
            'period_days': days_analyzed,
            'overall_assessment': overall,
            'overall_message': message,
            'total_anomalies': total_anomalies,
            'anomaly_rate_per_day': round(anomaly_rate, 2),
            'insights': insights,
            'generated_at': datetime.now().isoformat()
        }
