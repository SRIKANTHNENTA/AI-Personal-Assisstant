"""
Behavior Learner
Machine Learning module for analyzing user behavior patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from django.utils import timezone


class BehaviorLearner:
    """ML-based user behavior analysis"""
    
    def __init__(self):
        self.min_data_points = 10  # Minimum data points for analysis
    
    def detect_routine_patterns(self, user_activities: List[Dict]) -> Dict[str, any]:
        """
        Detect daily routine patterns from user activities
        """
        if len(user_activities) < self.min_data_points:
            return {
                'success': False,
                'error': 'Insufficient data for pattern detection',
                'patterns': []
            }
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(user_activities)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Detect morning routine (activities between 6-10 AM)
            morning_activities = df[(df['hour'] >= 6) & (df['hour'] <= 10)]
            morning_routine_time = None
            if len(morning_activities) > 0:
                morning_routine_time = morning_activities['hour'].mode()[0] if len(morning_activities['hour'].mode()) > 0 else None
            
            # Detect evening routine (activities between 6-10 PM)
            evening_activities = df[(df['hour'] >= 18) & (df['hour'] <= 22)]
            evening_routine_time = None
            if len(evening_activities) > 0:
                evening_routine_time = evening_activities['hour'].mode()[0] if len(evening_activities['hour'].mode()) > 0 else None
            
            # Find most active hours
            active_hours = df['hour'].value_counts().head(5).index.tolist()
            
            return {
                'success': True,
                'morning_routine_hour': morning_routine_time,
                'evening_routine_hour': evening_routine_time,
                'most_active_hours': active_hours,
                'total_activities_analyzed': len(df)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'patterns': []
            }
    
    def predict_task_completion_time(self, task_history: List[Dict], task_category: str) -> Dict[str, any]:
        """
        Predict task completion time based on historical data
        """
        if len(task_history) < 5:
            return {
                'success': False,
                'error': 'Insufficient historical data',
                'predicted_time': 30  # Default 30 minutes
            }
        
        try:
            # Filter by category
            category_tasks = [t for t in task_history if t.get('category') == task_category]
            
            if len(category_tasks) < 3:
                # Use all tasks if not enough in category
                category_tasks = task_history
            
            # Calculate average completion time
            completion_times = [t.get('completion_time', 30) for t in category_tasks]
            avg_time = np.mean(completion_times)
            std_time = np.std(completion_times)
            
            return {
                'success': True,
                'predicted_time': int(avg_time),
                'confidence': min(len(category_tasks) / 10, 1.0),  # Max confidence at 10 samples
                'time_range': {
                    'min': int(max(avg_time - std_time, 5)),
                    'max': int(avg_time + std_time)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predicted_time': 30
            }
    
    def analyze_task_completion_rate(self, tasks: List[Dict]) -> Dict[str, any]:
        """
        Analyze task completion patterns
        """
        if len(tasks) == 0:
            return {
                'success': False,
                'error': 'No tasks to analyze',
                'completion_rate': 0.0
            }
        
        try:
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.get('status') == 'completed'])
            on_time_tasks = len([t for t in tasks if t.get('on_time', False)])
            
            completion_rate = (completed_tasks / total_tasks) * 100
            on_time_rate = (on_time_tasks / completed_tasks * 100) if completed_tasks > 0 else 0
            
            # Analyze by category
            df = pd.DataFrame(tasks)
            category_stats = {}
            
            if 'category' in df.columns:
                for category in df['category'].unique():
                    cat_tasks = df[df['category'] == category]
                    cat_completed = len(cat_tasks[cat_tasks['status'] == 'completed'])
                    category_stats[category] = {
                        'total': len(cat_tasks),
                        'completed': cat_completed,
                        'completion_rate': (cat_completed / len(cat_tasks) * 100) if len(cat_tasks) > 0 else 0
                    }
            
            return {
                'success': True,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'completion_rate': completion_rate,
                'on_time_rate': on_time_rate,
                'category_stats': category_stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'completion_rate': 0.0
            }
    
    def detect_emotional_patterns(self, emotion_history: List[Dict]) -> Dict[str, any]:
        """
        Detect emotional patterns and stability
        """
        if len(emotion_history) < self.min_data_points:
            return {
                'success': False,
                'error': 'Insufficient emotional data',
                'dominant_emotion': 'neutral'
            }
        
        try:
            df = pd.DataFrame(emotion_history)
            
            # Find dominant emotion
            dominant_emotion = df['emotion'].mode()[0] if len(df['emotion'].mode()) > 0 else 'neutral'
            
            # Calculate emotional stability (lower variance = more stable)
            emotion_counts = df['emotion'].value_counts()
            total_count = len(df)
            entropy = -sum((count/total_count) * np.log2(count/total_count) for count in emotion_counts)
            
            # Normalize to 0-100 scale (lower entropy = more stable)
            max_entropy = np.log2(len(emotion_counts)) if len(emotion_counts) > 1 else 1
            stability_score = (1 - (entropy / max_entropy)) * 100 if max_entropy > 0 else 100
            
            # Calculate emotion distribution
            emotion_distribution = (emotion_counts / total_count * 100).to_dict()
            
            return {
                'success': True,
                'dominant_emotion': dominant_emotion,
                'emotional_stability_score': stability_score,
                'emotion_distribution': emotion_distribution,
                'total_records': total_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dominant_emotion': 'neutral'
            }
    
    def suggest_optimal_task_time(self, user_patterns: Dict, task_priority: str) -> Dict[str, any]:
        """
        Suggest optimal time for task based on user patterns
        """
        try:
            most_active_hours = user_patterns.get('most_active_hours', [9, 14, 16])
            
            # High priority tasks -> morning (most productive)
            if task_priority in ['high', 'urgent']:
                suggested_hour = most_active_hours[0] if most_active_hours else 9
            # Medium priority -> afternoon
            elif task_priority == 'medium':
                suggested_hour = most_active_hours[1] if len(most_active_hours) > 1 else 14
            # Low priority -> flexible
            else:
                suggested_hour = most_active_hours[-1] if most_active_hours else 16
            
            # Create suggested time (next occurrence of that hour)
            now = timezone.now()
            suggested_time = now.replace(hour=suggested_hour, minute=0, second=0, microsecond=0)
            
            if suggested_time < now:
                suggested_time += timedelta(days=1)
            
            return {
                'success': True,
                'suggested_time': suggested_time,
                'suggested_hour': suggested_hour,
                'reason': f'Based on your activity patterns, you are most productive around {suggested_hour}:00'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'suggested_time': timezone.now() + timedelta(hours=1)
            }
