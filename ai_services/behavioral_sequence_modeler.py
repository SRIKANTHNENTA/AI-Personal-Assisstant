"""
Behavioral Sequence Modeler
LSTM/Transformer-based behavioral sequence modeling for dynamic routine learning.
Predicts user activities, optimal task times, and behavioral patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import pickle


class BehavioralSequenceModeler:
    """
    LSTM/Transformer-based behavioral sequence modeling.
    Learns user routines dynamically and predicts future activities.
    """
    
    def __init__(self, user_id: str = None, sequence_length: int = 24):
        self.user_id = user_id
        self.sequence_length = sequence_length
        self.model = None
        self.activity_encoder = ActivityEncoder()
        self.time_encoder = TimeEncoder()
        
        # Model paths
        self.model_dir = Path(__file__).parent / 'behavioral_models'
        self.model_dir.mkdir(exist_ok=True)
        
        # Activity history
        self.activity_history = []
        self.max_history_size = 10000
        
        # Learned patterns
        self.temporal_patterns = defaultdict(list)
        self.sequence_patterns = defaultdict(int)
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        
        # Load existing model/data
        self._load_user_data()
    
    def _load_user_data(self):
        """Load user-specific behavioral data and model"""
        if self.user_id:
            data_path = self.model_dir / f'user_{self.user_id}_data.json'
            if data_path.exists():
                try:
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                        self.activity_history = data.get('activity_history', [])
                        self.temporal_patterns = defaultdict(list, data.get('temporal_patterns', {}))
                        self._rebuild_transition_matrix()
                except Exception as e:
                    print(f"Warning: Could not load user data: {e}")
    
    def _save_user_data(self):
        """Save user-specific behavioral data"""
        if self.user_id:
            data_path = self.model_dir / f'user_{self.user_id}_data.json'
            try:
                data = {
                    'activity_history': self.activity_history[-self.max_history_size:],
                    'temporal_patterns': dict(self.temporal_patterns),
                    'last_updated': datetime.now().isoformat()
                }
                with open(data_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save user data: {e}")
    
    def _rebuild_transition_matrix(self):
        """Rebuild activity transition matrix from history"""
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        
        for i in range(len(self.activity_history) - 1):
            current = self.activity_history[i].get('activity', 'unknown')
            next_act = self.activity_history[i + 1].get('activity', 'unknown')
            self.transition_matrix[current][next_act] += 1
        
        # Normalize to probabilities
        for activity, transitions in self.transition_matrix.items():
            total = sum(transitions.values())
            if total > 0:
                for next_act in transitions:
                    transitions[next_act] /= total
    
    def record_activity(self, activity: str, timestamp: datetime = None,
                       metadata: Dict = None) -> Dict:
        """
        Record a user activity for learning.
        
        Args:
            activity: Name of the activity (e.g., 'task_creation', 'chat', 'task_complete')
            timestamp: When the activity occurred (defaults to now)
            metadata: Additional context (duration, task_type, emotion, etc.)
            
        Returns:
            Dict with recording status and insights
        """
        timestamp = timestamp or datetime.now()
        
        # Create activity record
        record = {
            'activity': activity,
            'timestamp': timestamp.isoformat(),
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'metadata': metadata or {}
        }
        
        # Update temporal patterns
        time_key = f"{timestamp.weekday()}_{timestamp.hour}"
        self.temporal_patterns[time_key].append(activity)
        
        # Update transition matrix
        if self.activity_history:
            prev_activity = self.activity_history[-1].get('activity', 'unknown')
            self.transition_matrix[prev_activity][activity] += 1
            
            # Renormalize
            total = sum(self.transition_matrix[prev_activity].values())
            for act in self.transition_matrix[prev_activity]:
                self.transition_matrix[prev_activity][act] /= total
        
        # Update sequence patterns
        if len(self.activity_history) >= 2:
            recent_activities = [h['activity'] for h in self.activity_history[-2:]] + [activity]
            pattern_key = '|'.join(recent_activities)
            self.sequence_patterns[pattern_key] += 1
        
        # Add to history
        self.activity_history.append(record)
        
        # Check for routine detection
        routine_insight = self._detect_routine_formation(activity, timestamp)
        
        # Save periodically
        if len(self.activity_history) % 10 == 0:
            self._save_user_data()
        
        return {
            'success': True,
            'recorded': record,
            'routine_insight': routine_insight
        }
    
    def predict_next_activity(self, context: Dict = None) -> Dict:
        """
        Predict the next most likely activity.
        
        Args:
            context: Current context (time, recent activities, emotion)
            
        Returns:
            Dict with predicted activities and probabilities
        """
        if not self.activity_history:
            return {
                'success': False,
                'error': 'Insufficient activity history',
                'predictions': []
            }
        
        predictions = {}
        
        # 1. Temporal prediction (what usually happens at this time)
        now = datetime.now()
        time_key = f"{now.weekday()}_{now.hour}"
        
        if time_key in self.temporal_patterns:
            activities = self.temporal_patterns[time_key]
            activity_counts = defaultdict(int)
            for act in activities:
                activity_counts[act] += 1
            
            total = len(activities)
            for act, count in activity_counts.items():
                temporal_prob = count / total
                predictions[act] = predictions.get(act, 0) + 0.4 * temporal_prob
        
        # 2. Transition prediction (what usually follows the current activity)
        current_activity = self.activity_history[-1].get('activity', 'unknown')
        if current_activity in self.transition_matrix:
            for act, prob in self.transition_matrix[current_activity].items():
                predictions[act] = predictions.get(act, 0) + 0.4 * prob
        
        # 3. Sequence prediction (pattern matching)
        if len(self.activity_history) >= 2:
            recent = [h['activity'] for h in self.activity_history[-2:]]
            sequence_prefix = '|'.join(recent)
            
            matching_patterns = {
                k: v for k, v in self.sequence_patterns.items()
                if k.startswith(sequence_prefix)
            }
            
            if matching_patterns:
                total_matches = sum(matching_patterns.values())
                for pattern, count in matching_patterns.items():
                    next_act = pattern.split('|')[-1] if '|' in pattern else pattern
                    seq_prob = count / total_matches
                    predictions[next_act] = predictions.get(next_act, 0) + 0.2 * seq_prob
        
        if not predictions:
            return {
                'success': True,
                'predictions': [{'activity': 'unknown', 'probability': 0.0}],
                'method': 'no_data'
            }
        
        # Sort by probability
        sorted_predictions = sorted(
            [{'activity': act, 'probability': prob} for act, prob in predictions.items()],
            key=lambda x: x['probability'],
            reverse=True
        )[:5]
        
        return {
            'success': True,
            'predictions': sorted_predictions,
            'top_prediction': sorted_predictions[0] if sorted_predictions else None,
            'current_activity': current_activity,
            'method': 'hybrid_prediction'
        }
    
    def predict_optimal_task_time(self, task_type: str = None,
                                  priority: str = 'medium') -> Dict:
        """
        Predict optimal time to perform a task based on behavioral patterns.
        
        Args:
            task_type: Type of task (creative, analytical, routine, etc.)
            priority: Task priority level
            
        Returns:
            Dict with recommended time slots and confidence
        """
        if len(self.activity_history) < 20:
            return {
                'success': False,
                'error': 'Insufficient data for time prediction',
                'default_recommendation': {
                    'hour': 10,
                    'reason': 'Default productive hour (10 AM)'
                }
            }
        
        # Analyze task completion patterns
        completion_times = defaultdict(list)
        productivity_by_hour = defaultdict(list)
        
        for record in self.activity_history:
            if record.get('activity') == 'task_complete':
                metadata = record.get('metadata', {})
                hour = record.get('hour', 12)
                
                # Track completion success
                completion_times[hour].append({
                    'duration': metadata.get('duration', 30),
                    'on_time': metadata.get('on_time', True),
                    'task_type': metadata.get('task_type', 'unknown')
                })
        
        # Calculate productivity scores by hour
        hourly_scores = {}
        for hour in range(24):
            if hour in completion_times:
                completions = completion_times[hour]
                on_time_ratio = sum(1 for c in completions if c.get('on_time', True)) / len(completions)
                volume = len(completions)
                
                # Score combines on-time ratio and volume
                score = on_time_ratio * 0.7 + min(volume / 10, 1.0) * 0.3
                hourly_scores[hour] = score
        
        # Default scores for hours without data
        for hour in range(24):
            if hour not in hourly_scores:
                # Use circadian rhythm-based defaults
                if 9 <= hour <= 11:
                    hourly_scores[hour] = 0.7
                elif 14 <= hour <= 16:
                    hourly_scores[hour] = 0.6
                elif 6 <= hour <= 8:
                    hourly_scores[hour] = 0.5
                elif 20 <= hour <= 23:
                    hourly_scores[hour] = 0.4
                else:
                    hourly_scores[hour] = 0.3
        
        # Sort by score
        sorted_hours = sorted(hourly_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 recommendations
        recommendations = []
        for hour, score in sorted_hours[:3]:
            recommendations.append({
                'hour': hour,
                'time_range': f"{hour:02d}:00 - {(hour+1) % 24:02d}:00",
                'score': score,
                'confidence': min(len(completion_times.get(hour, [])) / 10, 1.0)
            })
        
        return {
            'success': True,
            'recommendations': recommendations,
            'best_time': recommendations[0] if recommendations else None,
            'analysis': {
                'hours_analyzed': len(hourly_scores),
                'total_completions': sum(len(v) for v in completion_times.values())
            }
        }
    
    def _detect_routine_formation(self, activity: str, timestamp: datetime) -> Optional[Dict]:
        """Detect if a routine is forming"""
        time_key = f"{timestamp.weekday()}_{timestamp.hour}"
        
        if time_key in self.temporal_patterns:
            activities = self.temporal_patterns[time_key]
            activity_count = activities.count(activity)
            
            # If same activity at same time at least 3 times
            if activity_count >= 3:
                consistency = activity_count / len(activities)
                if consistency > 0.5:
                    return {
                        'routine_detected': True,
                        'activity': activity,
                        'day': timestamp.strftime('%A'),
                        'hour': timestamp.hour,
                        'consistency': consistency,
                        'occurrences': activity_count,
                        'message': f"You often do '{activity}' at {timestamp.hour}:00 on {timestamp.strftime('%A')}s"
                    }
        
        return None
    
    def get_behavioral_summary(self) -> Dict:
        """
        Get a comprehensive behavioral summary.
        
        Returns:
            Dict with activity patterns, routines, and insights
        """
        if not self.activity_history:
            return {'success': False, 'error': 'No activity data'}
        
        # Activity frequency analysis
        activity_counts = defaultdict(int)
        hourly_activity = defaultdict(lambda: defaultdict(int))
        daily_activity = defaultdict(lambda: defaultdict(int))
        
        for record in self.activity_history:
            activity = record.get('activity', 'unknown')
            hour = record.get('hour', 12)
            day = record.get('day_of_week', 0)
            
            activity_counts[activity] += 1
            hourly_activity[hour][activity] += 1
            daily_activity[day][activity] += 1
        
        # Find most active hours
        hourly_totals = {hour: sum(acts.values()) for hour, acts in hourly_activity.items()}
        most_active_hours = sorted(hourly_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Find most common activities
        common_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Detect established routines
        routines = []
        for time_key, activities in self.temporal_patterns.items():
            if len(activities) >= 5:
                activity_counts_local = defaultdict(int)
                for act in activities:
                    activity_counts_local[act] += 1
                
                most_common = max(activity_counts_local.items(), key=lambda x: x[1])
                consistency = most_common[1] / len(activities)
                
                if consistency > 0.6:
                    day, hour = time_key.split('_')
                    routines.append({
                        'activity': most_common[0],
                        'day': int(day),
                        'hour': int(hour),
                        'consistency': consistency,
                        'occurrences': most_common[1]
                    })
        
        return {
            'success': True,
            'total_activities': len(self.activity_history),
            'unique_activities': len(activity_counts),
            'most_common_activities': common_activities,
            'most_active_hours': most_active_hours,
            'established_routines': sorted(routines, key=lambda x: x['consistency'], reverse=True)[:10],
            'activity_diversity': self._calculate_activity_diversity(),
            'behavior_consistency_score': self._calculate_consistency_score()
        }
    
    def _calculate_activity_diversity(self) -> float:
        """Calculate Shannon entropy for activity diversity"""
        if not self.activity_history:
            return 0.0
        
        activity_counts = defaultdict(int)
        for record in self.activity_history:
            activity_counts[record.get('activity', 'unknown')] += 1
        
        total = len(self.activity_history)
        entropy = 0.0
        for count in activity_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Normalize to 0-1
        max_entropy = np.log2(len(activity_counts)) if len(activity_counts) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_consistency_score(self) -> float:
        """Calculate how consistent user's behavior is"""
        if not self.temporal_patterns:
            return 0.5
        
        consistency_scores = []
        for time_key, activities in self.temporal_patterns.items():
            if len(activities) >= 3:
                activity_counts = defaultdict(int)
                for act in activities:
                    activity_counts[act] += 1
                
                max_count = max(activity_counts.values())
                consistency = max_count / len(activities)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5


class ActivityEncoder:
    """Encode activities to numeric representations"""
    
    def __init__(self):
        self.activity_to_id = {}
        self.id_to_activity = {}
        self.next_id = 0
    
    def encode(self, activity: str) -> int:
        if activity not in self.activity_to_id:
            self.activity_to_id[activity] = self.next_id
            self.id_to_activity[self.next_id] = activity
            self.next_id += 1
        return self.activity_to_id[activity]
    
    def decode(self, activity_id: int) -> str:
        return self.id_to_activity.get(activity_id, 'unknown')
    
    def encode_sequence(self, activities: List[str]) -> List[int]:
        return [self.encode(act) for act in activities]
    
    def decode_sequence(self, activity_ids: List[int]) -> List[str]:
        return [self.decode(aid) for aid in activity_ids]


class TimeEncoder:
    """Encode time features for sequence models"""
    
    def encode_timestamp(self, timestamp: datetime) -> np.ndarray:
        """
        Encode timestamp into feature vector.
        Uses cyclical encoding for periodic features.
        """
        hour = timestamp.hour
        minute = timestamp.minute
        day_of_week = timestamp.weekday()
        day_of_month = timestamp.day
        month = timestamp.month
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        return np.array([
            hour_sin, hour_cos,
            day_sin, day_cos,
            month_sin, month_cos,
            minute / 60.0,  # Normalized minute
            day_of_month / 31.0  # Normalized day of month
        ])


class LSTMBehaviorPredictor:
    """
    LSTM-based behavior prediction model.
    Requires TensorFlow/Keras to be installed.
    """
    
    def __init__(self, sequence_length: int = 24, n_features: int = 16):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.is_trained = False
        
        self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                layers.LSTM(64, input_shape=(self.sequence_length, self.n_features),
                           return_sequences=True, dropout=0.2),
                layers.LSTM(32, return_sequences=False, dropout=0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='softmax')  # Activity classes
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ LSTM Behavior Predictor model built successfully")
            
        except ImportError:
            print("⚠️ TensorFlow not installed. LSTM model disabled.")
            print("Install with: pip install tensorflow")
            self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model on behavioral sequences.
        
        Args:
            X: Input sequences (batch, sequence_length, features)
            y: Target activities (batch, n_classes)
            epochs: Training epochs
            batch_size: Training batch size
            
        Returns:
            Training history and metrics
        """
        if self.model is None:
            return {'success': False, 'error': 'Model not available'}
        
        try:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'success': True,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'val_loss': history.history.get('val_loss', [0])[-1],
                'val_accuracy': history.history.get('val_accuracy', [0])[-1]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next activity probabilities"""
        if self.model is None or not self.is_trained:
            return None
        
        return self.model.predict(X, verbose=0)


class TransformerBehaviorModel:
    """
    Transformer-based behavior prediction model.
    Uses self-attention for capturing long-range dependencies in behavior sequences.
    """
    
    def __init__(self, sequence_length: int = 48, d_model: int = 64, 
                 n_heads: int = 4, n_layers: int = 2):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model = None
        
        self._build_model()
    
    def _build_model(self):
        """Build Transformer model architecture"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            import tensorflow as tf
            
            # Input layer
            inputs = layers.Input(shape=(self.sequence_length, self.d_model))
            
            # Positional encoding
            x = self._add_positional_encoding(inputs)
            
            # Transformer blocks
            for _ in range(self.n_layers):
                x = self._transformer_block(x)
            
            # Output layers
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(16, activation='softmax')(x)
            
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Transformer Behavior Model built successfully")
            
        except ImportError:
            print("⚠️ TensorFlow not installed. Transformer model disabled.")
            self.model = None
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to input"""
        from tensorflow.keras import layers
        import tensorflow as tf
        
        seq_len = tf.shape(x)[1]
        d_model = tf.cast(self.d_model, tf.float32)
        
        positions = tf.range(self.sequence_length, dtype=tf.float32)
        dims = tf.range(self.d_model, dtype=tf.float32)
        
        angles = positions[:, tf.newaxis] / tf.pow(10000.0, 
                    (2 * (dims // 2)) / d_model)
        
        # Apply sin to even indices, cos to odd indices
        sin_mask = tf.cast(tf.range(self.d_model) % 2 == 0, tf.float32)
        cos_mask = 1 - sin_mask
        
        pos_encoding = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        pos_encoding = pos_encoding[tf.newaxis, :, :]
        
        return x + pos_encoding
    
    def _transformer_block(self, x):
        """Single transformer block with multi-head attention"""
        from tensorflow.keras import layers
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.n_heads, 
            key_dim=self.d_model // self.n_heads
        )(x, x)
        
        # Skip connection and layer norm
        x = layers.LayerNormalization()(x + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(self.d_model * 4, activation='relu')(x)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        
        # Skip connection and layer norm
        x = layers.LayerNormalization()(x + ffn_output)
        
        return x
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train the Transformer model"""
        if self.model is None:
            return {'success': False, 'error': 'Model not available'}
        
        try:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            return {
                'success': True,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next activity"""
        if self.model is None:
            return None
        return self.model.predict(X, verbose=0)


class BehaviorPatternMiner:
    """
    Mines behavioral patterns using association rules and sequence mining.
    Identifies frequently occurring activity sequences and correlations.
    """
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.patterns = []
        self.rules = []
    
    def mine_patterns(self, activity_sequences: List[List[str]]) -> Dict:
        """
        Mine frequent patterns from activity sequences.
        
        Args:
            activity_sequences: List of activity sequences
            
        Returns:
            Dict with patterns and association rules
        """
        if not activity_sequences:
            return {'success': False, 'error': 'No sequences provided'}
        
        # Count item frequencies
        item_counts = defaultdict(int)
        total_sequences = len(activity_sequences)
        
        for seq in activity_sequences:
            for item in set(seq):
                item_counts[item] += 1
        
        # Filter by minimum support
        frequent_items = {
            item: count / total_sequences 
            for item, count in item_counts.items() 
            if count / total_sequences >= self.min_support
        }
        
        # Mine sequential patterns (pairs)
        pair_counts = defaultdict(int)
        for seq in activity_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
        
        frequent_pairs = {}
        for pair, count in pair_counts.items():
            support = count / total_sequences
            if support >= self.min_support:
                # Calculate confidence
                ante_support = item_counts[pair[0]] / total_sequences
                confidence = support / ante_support if ante_support > 0 else 0
                
                if confidence >= self.min_confidence:
                    frequent_pairs[pair] = {
                        'support': support,
                        'confidence': confidence,
                        'lift': confidence / (item_counts[pair[1]] / total_sequences)
                        if item_counts[pair[1]] > 0 else 0
                    }
        
        self.patterns = frequent_pairs
        
        # Generate rules
        self.rules = [
            {
                'antecedent': pair[0],
                'consequent': pair[1],
                'support': metrics['support'],
                'confidence': metrics['confidence'],
                'lift': metrics['lift']
            }
            for pair, metrics in sorted(
                frequent_pairs.items(), 
                key=lambda x: x[1]['confidence'], 
                reverse=True
            )
        ]
        
        return {
            'success': True,
            'frequent_items': frequent_items,
            'frequent_pairs': len(frequent_pairs),
            'rules': self.rules[:20],  # Top 20 rules
            'total_sequences_analyzed': total_sequences
        }
    
    def predict_from_rules(self, current_activity: str) -> List[Dict]:
        """
        Predict next activity using mined rules.
        
        Args:
            current_activity: Current activity
            
        Returns:
            List of predictions with confidence
        """
        matching_rules = [
            rule for rule in self.rules 
            if rule['antecedent'] == current_activity
        ]
        
        return sorted(
            [
                {
                    'activity': rule['consequent'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift']
                }
                for rule in matching_rules
            ],
            key=lambda x: x['confidence'],
            reverse=True
        )
