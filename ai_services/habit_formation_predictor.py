"""
Habit Formation Predictor
AI-driven habit tracking and compliance prediction using behavioral data.
Predicts habit formation success and provides optimization recommendations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class HabitFormationPredictor:
    """
    Predicts habit formation success and provides compliance tracking.
    Uses behavioral patterns to optimize habit-building strategies.
    """
    
    # Habit formation stages (based on behavioral science)
    HABIT_STAGES = {
        'initiation': {'min_days': 0, 'max_days': 7, 'completion_threshold': 0.5},
        'learning': {'min_days': 8, 'max_days': 21, 'completion_threshold': 0.7},
        'stability': {'min_days': 22, 'max_days': 66, 'completion_threshold': 0.8},
        'maintenance': {'min_days': 67, 'max_days': None, 'completion_threshold': 0.85}
    }
    
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        
        # Habit tracking data
        self.habits: Dict[str, Dict] = {}
        self.completion_logs: Dict[str, List] = defaultdict(list)
        
        # ML models
        self.success_predictor = None
        self.days_to_form_predictor = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.models_trained = False
        
        # User behavior context
        self.user_context = {
            'typical_active_hours': list(range(8, 22)),
            'high_energy_times': [9, 10, 14, 15],
            'low_energy_times': [13, 14, 21, 22],
            'weekday_vs_weekend_compliance': 1.0
        }
        
        # Model persistence
        self.model_dir = Path(__file__).parent / 'habit_models'
        self.model_dir.mkdir(exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load saved habit data"""
        if self.user_id:
            path = self.model_dir / f'habits_{self.user_id}.json'
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        self.habits = data.get('habits', {})
                        self.completion_logs = defaultdict(list, data.get('completion_logs', {}))
                        self.user_context = data.get('user_context', self.user_context)
                except Exception as e:
                    print(f"Warning: Could not load habit data: {e}")
    
    def _save_data(self):
        """Save habit data"""
        if self.user_id:
            path = self.model_dir / f'habits_{self.user_id}.json'
            try:
                data = {
                    'habits': self.habits,
                    'completion_logs': dict(self.completion_logs),
                    'user_context': self.user_context,
                    'last_updated': datetime.now().isoformat()
                }
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save habit data: {e}")
    
    def create_habit(self, 
                    habit_name: str,
                    category: str = 'general',
                    target_frequency: str = 'daily',
                    target_times: List[int] = None,
                    difficulty: float = 0.5,
                    motivation: str = None,
                    cue: str = None,
                    reward: str = None) -> Dict:
        """
        Create a new habit to track.
        
        Args:
            habit_name: Name of the habit
            category: Category (health, productivity, learning, etc.)
            target_frequency: daily, weekly, specific_days
            target_times: Preferred times of day (hours 0-23)
            difficulty: Perceived difficulty 0-1
            motivation: Why forming this habit
            cue: Trigger/cue for the habit
            reward: Reward after completion
            
        Returns:
            Dict with habit creation details
        """
        habit_id = f"habit_{len(self.habits) + 1}_{int(datetime.now().timestamp())}"
        
        self.habits[habit_id] = {
            'id': habit_id,
            'name': habit_name,
            'category': category,
            'frequency': target_frequency,
            'target_times': target_times or [9, 18],  # Default morning/evening
            'difficulty': difficulty,
            'motivation': motivation,
            'cue': cue,
            'reward': reward,
            'created_at': datetime.now().isoformat(),
            'start_date': datetime.now().date().isoformat(),
            'status': 'active',
            'current_streak': 0,
            'longest_streak': 0,
            'total_completions': 0,
            'stage': 'initiation'
        }
        
        # Initialize prediction
        prediction = self.predict_habit_success(habit_id)
        
        self._save_data()
        
        return {
            'success': True,
            'habit_id': habit_id,
            'habit': self.habits[habit_id],
            'initial_prediction': prediction
        }
    
    def log_completion(self,
                      habit_id: str,
                      completed: bool = True,
                      completion_time: datetime = None,
                      difficulty_felt: float = None,
                      notes: str = None) -> Dict:
        """
        Log a habit completion or miss.
        
        Args:
            habit_id: ID of the habit
            completed: Whether the habit was completed
            completion_time: When it was completed
            difficulty_felt: Subjective difficulty 0-1
            notes: Any notes about the completion
            
        Returns:
            Updated habit stats and predictions
        """
        if habit_id not in self.habits:
            return {'success': False, 'error': 'Habit not found'}
        
        completion_time = completion_time or datetime.now()
        
        log_entry = {
            'timestamp': completion_time.isoformat(),
            'date': completion_time.date().isoformat(),
            'hour': completion_time.hour,
            'day_of_week': completion_time.weekday(),
            'completed': completed,
            'difficulty_felt': difficulty_felt,
            'notes': notes
        }
        
        self.completion_logs[habit_id].append(log_entry)
        
        # Update habit stats
        habit = self.habits[habit_id]
        if completed:
            habit['total_completions'] += 1
            habit['current_streak'] = self._calculate_streak(habit_id)
            habit['longest_streak'] = max(habit['longest_streak'], habit['current_streak'])
        else:
            habit['current_streak'] = 0
        
        # Update stage
        days_active = self._days_since_start(habit_id)
        habit['stage'] = self._determine_stage(days_active)
        
        # Update user context
        if completed:
            self._update_user_context(completion_time)
        
        self._save_data()
        
        # Get updated predictions
        stats = self.get_habit_stats(habit_id)
        prediction = self.predict_habit_success(habit_id)
        
        return {
            'success': True,
            'logged': log_entry,
            'current_stats': stats,
            'prediction': prediction,
            'recommendations': self.get_optimization_recommendations(habit_id)
        }
    
    def _calculate_streak(self, habit_id: str) -> int:
        """Calculate current completion streak"""
        logs = self.completion_logs.get(habit_id, [])
        if not logs:
            return 0
        
        # Get unique completion dates in descending order
        completion_dates = sorted(
            set(log['date'] for log in logs if log['completed']),
            reverse=True
        )
        
        if not completion_dates:
            return 0
        
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        
        # Check if streak is current
        if completion_dates[0] != today and completion_dates[0] != yesterday:
            return 0
        
        streak = 1
        for i in range(1, len(completion_dates)):
            current = datetime.fromisoformat(completion_dates[i - 1])
            previous = datetime.fromisoformat(completion_dates[i])
            if (current - previous).days == 1:
                streak += 1
            else:
                break
        
        return streak
    
    def _days_since_start(self, habit_id: str) -> int:
        """Calculate days since habit was created"""
        habit = self.habits.get(habit_id)
        if not habit:
            return 0
        
        start_date = datetime.fromisoformat(habit['start_date']).date()
        return (datetime.now().date() - start_date).days
    
    def _determine_stage(self, days_active: int) -> str:
        """Determine habit formation stage"""
        for stage, params in self.HABIT_STAGES.items():
            min_days = params['min_days']
            max_days = params['max_days']
            
            if max_days is None:
                if days_active >= min_days:
                    return stage
            elif min_days <= days_active <= max_days:
                return stage
        
        return 'initiation'
    
    def _update_user_context(self, completion_time: datetime):
        """Update user behavior context based on completion"""
        hour = completion_time.hour
        is_weekend = completion_time.weekday() >= 5
        
        # Track active hours
        if hour not in self.user_context['typical_active_hours']:
            self.user_context['typical_active_hours'].append(hour)
    
    def get_habit_stats(self, habit_id: str) -> Dict:
        """Get comprehensive habit statistics"""
        if habit_id not in self.habits:
            return {'success': False, 'error': 'Habit not found'}
        
        habit = self.habits[habit_id]
        logs = self.completion_logs.get(habit_id, [])
        
        if not logs:
            return {
                'success': True,
                'habit_id': habit_id,
                'name': habit['name'],
                'total_logs': 0,
                'completion_rate': 0,
                'current_streak': 0,
                'stage': habit['stage'],
                'days_active': self._days_since_start(habit_id)
            }
        
        completed_logs = [l for l in logs if l['completed']]
        
        # Calculate completion rate
        days_active = max(1, self._days_since_start(habit_id))
        if habit['frequency'] == 'daily':
            expected_completions = days_active
        else:
            expected_completions = len(logs)  # Simplified
        
        completion_rate = len(completed_logs) / max(1, expected_completions)
        
        # Time analysis
        completion_hours = [l['hour'] for l in completed_logs]
        preferred_hour = max(set(completion_hours), key=completion_hours.count) if completion_hours else None
        
        # Day of week analysis
        completion_days = [l['day_of_week'] for l in completed_logs]
        best_day = max(set(completion_days), key=completion_days.count) if completion_days else None
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Difficulty analysis
        difficulties = [l['difficulty_felt'] for l in logs if l.get('difficulty_felt') is not None]
        avg_difficulty = np.mean(difficulties) if difficulties else None
        
        return {
            'success': True,
            'habit_id': habit_id,
            'name': habit['name'],
            'category': habit['category'],
            'total_logs': len(logs),
            'total_completions': len(completed_logs),
            'completion_rate': round(completion_rate, 3),
            'current_streak': habit['current_streak'],
            'longest_streak': habit['longest_streak'],
            'stage': habit['stage'],
            'days_active': days_active,
            'preferred_hour': preferred_hour,
            'best_day': day_names[best_day] if best_day is not None else None,
            'average_difficulty': round(avg_difficulty, 2) if avg_difficulty else None,
            'stage_threshold': self.HABIT_STAGES[habit['stage']]['completion_threshold']
        }
    
    def predict_habit_success(self, habit_id: str) -> Dict:
        """
        Predict likelihood of habit formation success.
        
        Args:
            habit_id: ID of the habit
            
        Returns:
            Prediction results with confidence
        """
        if habit_id not in self.habits:
            return {'success': False, 'error': 'Habit not found'}
        
        habit = self.habits[habit_id]
        stats = self.get_habit_stats(habit_id)
        
        # Feature extraction
        features = self._extract_features(habit_id)
        
        # Use ML model if trained and available
        if self.models_trained and ML_AVAILABLE and self.success_predictor:
            try:
                scaled_features = self.scaler.transform([list(features.values())])
                success_prob = self.success_predictor.predict_proba(scaled_features)[0][1]
            except Exception:
                success_prob = self._heuristic_success_prediction(features, stats)
        else:
            success_prob = self._heuristic_success_prediction(features, stats)
        
        # Estimate days to habit formation
        days_to_form = self._estimate_days_to_form(features, success_prob)
        
        # Risk factors
        risk_factors = self._identify_risk_factors(features, stats)
        
        # Confidence based on data availability
        confidence = min(1.0, stats['days_active'] / 14)  # Max confidence after 2 weeks
        
        return {
            'success': True,
            'habit_id': habit_id,
            'success_probability': round(success_prob, 3),
            'confidence': round(confidence, 3),
            'estimated_days_to_form': days_to_form,
            'risk_factors': risk_factors,
            'current_stage': habit['stage'],
            'stage_progress': self._calculate_stage_progress(habit_id)
        }
    
    def _extract_features(self, habit_id: str) -> Dict:
        """Extract features for ML prediction"""
        habit = self.habits.get(habit_id, {})
        logs = self.completion_logs.get(habit_id, [])
        stats = self.get_habit_stats(habit_id)
        
        # Basic features
        features = {
            'difficulty': habit.get('difficulty', 0.5),
            'days_active': stats.get('days_active', 0),
            'completion_rate': stats.get('completion_rate', 0),
            'current_streak': stats.get('current_streak', 0),
            'longest_streak': stats.get('longest_streak', 0),
            'total_completions': stats.get('total_completions', 0),
        }
        
        # Consistency features
        if len(logs) >= 7:
            recent_logs = logs[-7:]
            recent_completions = sum(1 for l in recent_logs if l['completed'])
            features['recent_7day_rate'] = recent_completions / 7
        else:
            features['recent_7day_rate'] = features['completion_rate']
        
        # Time consistency
        if logs:
            completion_hours = [l['hour'] for l in logs if l['completed']]
            if completion_hours:
                features['hour_variance'] = np.var(completion_hours) / 144  # Normalize by max variance
            else:
                features['hour_variance'] = 1.0
        else:
            features['hour_variance'] = 0.5
        
        # Trend feature
        features['trend'] = self._calculate_trend(habit_id)
        
        # Has cue/reward
        features['has_cue'] = 1.0 if habit.get('cue') else 0.0
        features['has_reward'] = 1.0 if habit.get('reward') else 0.0
        
        return features
    
    def _heuristic_success_prediction(self, features: Dict, stats: Dict) -> float:
        """Heuristic-based success prediction when ML unavailable"""
        score = 0.5  # Base score
        
        # Completion rate impact
        completion_rate = features.get('completion_rate', 0)
        score += (completion_rate - 0.5) * 0.3
        
        # Streak impact
        streak = features.get('current_streak', 0)
        if streak > 7:
            score += 0.15
        elif streak > 3:
            score += 0.08
        elif streak == 0:
            score -= 0.1
        
        # Consistency impact
        hour_variance = features.get('hour_variance', 0.5)
        if hour_variance < 0.1:
            score += 0.1  # Consistent timing
        
        # Difficulty impact
        difficulty = features.get('difficulty', 0.5)
        score -= (difficulty - 0.5) * 0.2
        
        # Cue/reward impact
        if features.get('has_cue'):
            score += 0.05
        if features.get('has_reward'):
            score += 0.05
        
        # Trend impact
        trend = features.get('trend', 0)
        score += trend * 0.15
        
        # Stage bonus
        stage = stats.get('stage', 'initiation')
        stage_bonus = {'initiation': 0, 'learning': 0.1, 'stability': 0.2, 'maintenance': 0.3}
        score += stage_bonus.get(stage, 0)
        
        return np.clip(score, 0.05, 0.95)
    
    def _calculate_trend(self, habit_id: str) -> float:
        """Calculate completion trend (-1 to 1)"""
        logs = self.completion_logs.get(habit_id, [])
        
        if len(logs) < 4:
            return 0.0
        
        # Compare first half vs second half
        mid = len(logs) // 2
        first_half = logs[:mid]
        second_half = logs[mid:]
        
        first_rate = sum(1 for l in first_half if l['completed']) / len(first_half)
        second_rate = sum(1 for l in second_half if l['completed']) / len(second_half)
        
        return second_rate - first_rate
    
    def _estimate_days_to_form(self, features: Dict, success_prob: float) -> int:
        """Estimate days needed to form the habit"""
        base_days = 66  # Research average
        
        # Adjust based on difficulty
        difficulty = features.get('difficulty', 0.5)
        difficulty_factor = 1 + (difficulty - 0.5)
        
        # Adjust based on current progress
        completion_rate = features.get('completion_rate', 0)
        progress_factor = 1 - (completion_rate * 0.3)
        
        # Adjust based on success probability
        prob_factor = 1.5 - success_prob
        
        estimated = base_days * difficulty_factor * progress_factor * prob_factor
        
        # Subtract days already active
        days_active = features.get('days_active', 0)
        remaining = int(estimated - days_active)
        
        return max(7, remaining)  # Minimum 7 days
    
    def _identify_risk_factors(self, features: Dict, stats: Dict) -> List[Dict]:
        """Identify risk factors for habit failure"""
        risks = []
        
        # Low completion rate
        if features['completion_rate'] < 0.5:
            risks.append({
                'factor': 'low_completion_rate',
                'severity': 'high' if features['completion_rate'] < 0.3 else 'medium',
                'message': 'Completion rate is below target',
                'recommendation': 'Consider simplifying the habit or adjusting timing'
            })
        
        # Broken streak
        if features['longest_streak'] > 5 and features['current_streak'] == 0:
            risks.append({
                'factor': 'streak_broken',
                'severity': 'medium',
                'message': 'Previous streak was broken',
                'recommendation': 'Focus on rebuilding momentum with small wins'
            })
        
        # High difficulty
        if features['difficulty'] > 0.7:
            risks.append({
                'factor': 'high_difficulty',
                'severity': 'medium',
                'message': 'Habit perceived as difficult',
                'recommendation': 'Break into smaller steps or reduce scope initially'
            })
        
        # Inconsistent timing
        if features['hour_variance'] > 0.3:
            risks.append({
                'factor': 'inconsistent_timing',
                'severity': 'low',
                'message': 'Habit timing varies significantly',
                'recommendation': 'Set a specific time each day for better adherence'
            })
        
        # Negative trend
        if features['trend'] < -0.2:
            risks.append({
                'factor': 'declining_trend',
                'severity': 'high',
                'message': 'Completion rate is declining',
                'recommendation': 'Revisit motivation or reduce habit complexity'
            })
        
        # No cue defined
        if not features['has_cue']:
            risks.append({
                'factor': 'no_cue',
                'severity': 'low',
                'message': 'No trigger/cue defined',
                'recommendation': 'Define a clear trigger (e.g., "After breakfast...")'
            })
        
        # No reward defined
        if not features['has_reward']:
            risks.append({
                'factor': 'no_reward',
                'severity': 'low',
                'message': 'No reward defined',
                'recommendation': 'Define a reward to reinforce the behavior'
            })
        
        return risks
    
    def _calculate_stage_progress(self, habit_id: str) -> Dict:
        """Calculate progress within current stage"""
        habit = self.habits.get(habit_id, {})
        stage = habit.get('stage', 'initiation')
        stats = self.get_habit_stats(habit_id)
        
        stage_info = self.HABIT_STAGES.get(stage, self.HABIT_STAGES['initiation'])
        
        days_active = stats.get('days_active', 0)
        min_days = stage_info['min_days']
        max_days = stage_info['max_days'] or (min_days + 30)
        
        # Days progress
        days_in_stage = days_active - min_days
        total_days_in_stage = max_days - min_days
        days_progress = min(1.0, days_in_stage / total_days_in_stage)
        
        # Completion threshold progress
        current_rate = stats.get('completion_rate', 0)
        threshold = stage_info['completion_threshold']
        threshold_progress = min(1.0, current_rate / threshold)
        
        # Combined progress
        combined_progress = (days_progress + threshold_progress) / 2
        
        return {
            'stage': stage,
            'days_progress': round(days_progress, 2),
            'threshold_progress': round(threshold_progress, 2),
            'combined_progress': round(combined_progress, 2),
            'threshold_required': threshold,
            'current_completion_rate': current_rate
        }
    
    def get_optimization_recommendations(self, habit_id: str) -> List[Dict]:
        """
        Get personalized recommendations for optimizing habit success.
        
        Args:
            habit_id: ID of the habit
            
        Returns:
            List of actionable recommendations
        """
        if habit_id not in self.habits:
            return []
        
        habit = self.habits[habit_id]
        stats = self.get_habit_stats(habit_id)
        prediction = self.predict_habit_success(habit_id)
        
        recommendations = []
        
        # Time optimization
        if stats.get('preferred_hour'):
            target_times = habit.get('target_times', [])
            if stats['preferred_hour'] not in target_times:
                recommendations.append({
                    'type': 'timing',
                    'priority': 'medium',
                    'title': 'Optimize timing',
                    'description': f"You tend to complete this habit at {stats['preferred_hour']}:00. "
                                 f"Consider scheduling it at this time.",
                    'action': f"Update target time to {stats['preferred_hour']}:00"
                })
        
        # Streak building
        if habit['current_streak'] >= 3 and habit['current_streak'] < 7:
            recommendations.append({
                'type': 'motivation',
                'priority': 'high',
                'title': 'Keep the streak!',
                'description': f"You're on a {habit['current_streak']}-day streak! "
                             f"Keep going to build momentum.",
                'action': 'Focus on maintaining your streak for the next few days'
            })
        
        # Stage-specific advice
        stage = habit['stage']
        if stage == 'initiation':
            recommendations.append({
                'type': 'strategy',
                'priority': 'high',
                'title': 'Start small',
                'description': 'In the first week, focus on consistency over perfection. '
                             'Even partial completion counts!',
                'action': 'Reduce the habit to its smallest viable version if needed'
            })
        elif stage == 'learning':
            recommendations.append({
                'type': 'strategy',
                'priority': 'medium',
                'title': 'Build the routine',
                'description': 'Now is the time to solidify the habit loop. '
                             'Ensure you have a clear cue and reward.',
                'action': 'Strengthen your cue-routine-reward connection'
            })
        elif stage == 'stability':
            recommendations.append({
                'type': 'strategy',
                'priority': 'medium',
                'title': 'Prepare for challenges',
                'description': 'Your habit is stabilizing. Plan for disruptions '
                             '(travel, stress, schedule changes).',
                'action': 'Create a backup plan for missed days'
            })
        
        # Cue recommendation
        if not habit.get('cue'):
            recommendations.append({
                'type': 'setup',
                'priority': 'high',
                'title': 'Add a trigger cue',
                'description': 'Habits form faster with clear triggers. '
                             'Link this habit to an existing behavior.',
                'action': 'Add a cue like "After I [existing habit], I will [new habit]"'
            })
        
        # Reward recommendation  
        if not habit.get('reward'):
            recommendations.append({
                'type': 'setup',
                'priority': 'medium',
                'title': 'Define a reward',
                'description': 'Rewards reinforce the habit loop. Even small rewards help.',
                'action': 'Add a reward you can give yourself after completion'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 1))
        
        return recommendations
    
    def get_daily_habit_schedule(self) -> List[Dict]:
        """
        Get today's habit schedule with optimal ordering.
        
        Returns:
            Ordered list of habits for today
        """
        now = datetime.now()
        today_habits = []
        
        for habit_id, habit in self.habits.items():
            if habit['status'] != 'active':
                continue
            
            # Check if due today
            if habit['frequency'] == 'daily':
                is_due = True
            else:
                is_due = True  # Simplified
            
            if not is_due:
                continue
            
            # Check if already completed today
            logs_today = [
                l for l in self.completion_logs.get(habit_id, [])
                if l['date'] == now.date().isoformat() and l['completed']
            ]
            
            stats = self.get_habit_stats(habit_id)
            prediction = self.predict_habit_success(habit_id)
            
            today_habits.append({
                'habit_id': habit_id,
                'name': habit['name'],
                'category': habit['category'],
                'target_times': habit.get('target_times', [9]),
                'completed_today': len(logs_today) > 0,
                'current_streak': habit['current_streak'],
                'success_probability': prediction.get('success_probability', 0.5),
                'priority_score': self._calculate_priority_score(habit, stats)
            })
        
        # Sort by target time, then priority
        today_habits.sort(key=lambda x: (
            min(x['target_times']) if x['target_times'] else 12,
            -x['priority_score']
        ))
        
        return today_habits
    
    def _calculate_priority_score(self, habit: Dict, stats: Dict) -> float:
        """Calculate habit priority score for scheduling"""
        score = 0.5
        
        # Streak at risk bonus
        if habit['current_streak'] >= 3:
            score += 0.2
        
        # Struggling habits need attention
        completion_rate = stats.get('completion_rate', 0.5)
        if completion_rate < 0.5:
            score += 0.15
        
        # Stage consideration
        if habit['stage'] == 'initiation':
            score += 0.1  # New habits need extra focus
        
        return score
    
    def train_models(self, external_data: List[Dict] = None) -> Dict:
        """
        Train ML models on habit completion data.
        
        Args:
            external_data: Optional external training data
            
        Returns:
            Training results
        """
        if not ML_AVAILABLE:
            return {'success': False, 'error': 'ML libraries not available'}
        
        # Collect training data from all habits
        X, y = [], []
        
        for habit_id, habit in self.habits.items():
            if len(self.completion_logs.get(habit_id, [])) < 7:
                continue  # Need minimum data
            
            features = self._extract_features(habit_id)
            
            # Label: successful if completion rate > threshold
            threshold = self.HABIT_STAGES[habit['stage']]['completion_threshold']
            success = 1 if features['completion_rate'] >= threshold else 0
            
            X.append(list(features.values()))
            y.append(success)
        
        if len(X) < 5:
            return {'success': False, 'error': 'Insufficient training data (need at least 5 habits)'}
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train success predictor
            self.success_predictor = RandomForestClassifier(n_estimators=50, random_state=42)
            self.success_predictor.fit(X_scaled, y)
            
            self.models_trained = True
            
            return {
                'success': True,
                'training_samples': len(X),
                'positive_samples': sum(y),
                'model_type': 'RandomForestClassifier'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_all_habits_summary(self) -> Dict:
        """Get summary of all tracked habits"""
        if not self.habits:
            return {
                'total_habits': 0,
                'active_habits': 0,
                'habits': []
            }
        
        habits_summary = []
        for habit_id in self.habits:
            stats = self.get_habit_stats(habit_id)
            prediction = self.predict_habit_success(habit_id)
            habits_summary.append({
                'habit_id': habit_id,
                'name': self.habits[habit_id]['name'],
                'stage': self.habits[habit_id]['stage'],
                'current_streak': self.habits[habit_id]['current_streak'],
                'completion_rate': stats.get('completion_rate', 0),
                'success_probability': prediction.get('success_probability', 0.5)
            })
        
        active_habits = [h for h in self.habits.values() if h['status'] == 'active']
        
        return {
            'total_habits': len(self.habits),
            'active_habits': len(active_habits),
            'average_completion_rate': np.mean([h['completion_rate'] for h in habits_summary]) if habits_summary else 0,
            'habits': habits_summary
        }
