"""
Cognitive Analytics Dashboard
Provides visual analytics data for user emotional and productivity trends.
Generates chart-ready data structures and comprehensive reports.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path


class CognitiveAnalyticsDashboard:
    """
    Generates comprehensive analytics data for cognitive and productivity insights.
    Provides chart-ready data structures for frontend visualization.
    """
    
    # Time periods for analysis
    TIME_PERIODS = {
        'day': timedelta(days=1),
        'week': timedelta(days=7),
        'month': timedelta(days=30),
        'quarter': timedelta(days=90)
    }
    
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        
        # Data stores
        self.emotion_data: List[Dict] = []
        self.productivity_data: List[Dict] = []
        self.task_data: List[Dict] = []
        self.interaction_data: List[Dict] = []
        self.habit_data: List[Dict] = []
        
        # Computed metrics cache
        self._metrics_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=5)
        
        # Model persistence
        self.data_dir = Path(__file__).parent / 'analytics_data'
        self.data_dir.mkdir(exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load saved analytics data"""
        if self.user_id:
            path = self.data_dir / f'analytics_{self.user_id}.json'
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        self.emotion_data = data.get('emotion_data', [])[-5000:]
                        self.productivity_data = data.get('productivity_data', [])[-5000:]
                        self.task_data = data.get('task_data', [])[-5000:]
                        self.interaction_data = data.get('interaction_data', [])[-5000:]
                        self.habit_data = data.get('habit_data', [])[-5000:]
                except Exception as e:
                    print(f"Warning: Could not load analytics data: {e}")
    
    def _save_data(self):
        """Save analytics data"""
        if self.user_id:
            path = self.data_dir / f'analytics_{self.user_id}.json'
            try:
                data = {
                    'emotion_data': self.emotion_data[-5000:],
                    'productivity_data': self.productivity_data[-5000:],
                    'task_data': self.task_data[-5000:],
                    'interaction_data': self.interaction_data[-5000:],
                    'habit_data': self.habit_data[-5000:],
                    'last_updated': datetime.now().isoformat()
                }
                with open(path, 'w') as f:
                    json.dump(data, f, default=str)
            except Exception as e:
                print(f"Warning: Could not save analytics data: {e}")
    
    def ingest_emotion_data(self, data: Dict) -> bool:
        """
        Ingest emotion data point.
        
        Args:
            data: Dict with emotion, valence, arousal, timestamp, source
            
        Returns:
            Success status
        """
        try:
            entry = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'emotion': data.get('emotion', 'neutral'),
                'valence': data.get('valence', 0),
                'arousal': data.get('arousal', 0.5),
                'confidence': data.get('confidence', 0.8),
                'source': data.get('source', 'unknown')
            }
            self.emotion_data.append(entry)
            self._invalidate_cache()
            
            if len(self.emotion_data) % 100 == 0:
                self._save_data()
            
            return True
        except Exception:
            return False
    
    def ingest_productivity_data(self, data: Dict) -> bool:
        """
        Ingest productivity data point.
        
        Args:
            data: Dict with productivity_score, focus_time, tasks_completed, etc.
        """
        try:
            entry = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'productivity_score': data.get('productivity_score', 0.5),
                'focus_time_minutes': data.get('focus_time_minutes', 0),
                'tasks_completed': data.get('tasks_completed', 0),
                'tasks_started': data.get('tasks_started', 0),
                'distractions': data.get('distractions', 0),
                'context': data.get('context', 'general')
            }
            self.productivity_data.append(entry)
            self._invalidate_cache()
            return True
        except Exception:
            return False
    
    def ingest_task_data(self, data: Dict) -> bool:
        """Ingest task completion data"""
        try:
            entry = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'task_id': data.get('task_id'),
                'task_name': data.get('task_name'),
                'priority': data.get('priority', 'medium'),
                'category': data.get('category', 'general'),
                'estimated_duration': data.get('estimated_duration'),
                'actual_duration': data.get('actual_duration'),
                'completed': data.get('completed', True),
                'completion_quality': data.get('completion_quality', 0.8)
            }
            self.task_data.append(entry)
            self._invalidate_cache()
            return True
        except Exception:
            return False
    
    def ingest_habit_data(self, data: Dict) -> bool:
        """Ingest habit tracking data"""
        try:
            entry = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'habit_id': data.get('habit_id'),
                'habit_name': data.get('habit_name'),
                'completed': data.get('completed', True),
                'streak': data.get('streak', 0),
                'difficulty': data.get('difficulty', 0.5)
            }
            self.habit_data.append(entry)
            self._invalidate_cache()
            return True
        except Exception:
            return False
    
    def _invalidate_cache(self):
        """Invalidate metrics cache"""
        self._cache_timestamp = None
        self._metrics_cache = {}
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration
    
    def get_emotion_trends(self, 
                          period: str = 'week',
                          granularity: str = 'day') -> Dict:
        """
        Get emotion trends over time.
        
        Args:
            period: Time period ('day', 'week', 'month', 'quarter')
            granularity: Data granularity ('hour', 'day', 'week')
            
        Returns:
            Chart-ready emotion trend data
        """
        cache_key = f'emotion_trends_{period}_{granularity}'
        if self._is_cache_valid() and cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]
        
        cutoff = datetime.now() - self.TIME_PERIODS.get(period, timedelta(days=7))
        
        filtered = [
            e for e in self.emotion_data
            if datetime.fromisoformat(e['timestamp']) > cutoff
        ]
        
        if not filtered:
            return {
                'success': False,
                'error': 'No data available for this period',
                'period': period
            }
        
        # Group by granularity
        grouped = self._group_by_time(filtered, granularity)
        
        # Calculate metrics per group
        series_valence = []
        series_arousal = []
        series_emotion_distribution = []
        labels = []
        
        for label, entries in sorted(grouped.items()):
            labels.append(label)
            
            valences = [e['valence'] for e in entries]
            arousals = [e['arousal'] for e in entries]
            emotions = [e['emotion'] for e in entries]
            
            series_valence.append({
                'x': label,
                'y': round(np.mean(valences), 3),
                'min': round(min(valences), 3),
                'max': round(max(valences), 3)
            })
            
            series_arousal.append({
                'x': label,
                'y': round(np.mean(arousals), 3)
            })
            
            # Emotion distribution
            emotion_counts = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            series_emotion_distribution.append({
                'x': label,
                'distribution': emotion_counts
            })
        
        result = {
            'success': True,
            'period': period,
            'granularity': granularity,
            'data_points': len(filtered),
            'labels': labels,
            'series': {
                'valence': series_valence,
                'arousal': series_arousal,
                'emotion_distribution': series_emotion_distribution
            },
            'summary': {
                'avg_valence': round(np.mean([e['valence'] for e in filtered]), 3),
                'avg_arousal': round(np.mean([e['arousal'] for e in filtered]), 3),
                'dominant_emotion': max(set([e['emotion'] for e in filtered]), 
                                       key=[e['emotion'] for e in filtered].count)
            }
        }
        
        self._metrics_cache[cache_key] = result
        self._cache_timestamp = datetime.now()
        
        return result
    
    def get_productivity_trends(self,
                               period: str = 'week',
                               granularity: str = 'day') -> Dict:
        """
        Get productivity trends over time.
        
        Args:
            period: Time period
            granularity: Data granularity
            
        Returns:
            Chart-ready productivity data
        """
        cutoff = datetime.now() - self.TIME_PERIODS.get(period, timedelta(days=7))
        
        filtered = [
            p for p in self.productivity_data
            if datetime.fromisoformat(p['timestamp']) > cutoff
        ]
        
        if not filtered:
            return {
                'success': False,
                'error': 'No productivity data available',
                'period': period
            }
        
        grouped = self._group_by_time(filtered, granularity)
        
        series_score = []
        series_focus = []
        series_tasks = []
        labels = []
        
        for label, entries in sorted(grouped.items()):
            labels.append(label)
            
            scores = [e['productivity_score'] for e in entries]
            focus_times = [e['focus_time_minutes'] for e in entries]
            tasks = sum(e['tasks_completed'] for e in entries)
            
            series_score.append({
                'x': label,
                'y': round(np.mean(scores), 3)
            })
            
            series_focus.append({
                'x': label,
                'y': sum(focus_times)
            })
            
            series_tasks.append({
                'x': label,
                'y': tasks
            })
        
        return {
            'success': True,
            'period': period,
            'granularity': granularity,
            'data_points': len(filtered),
            'labels': labels,
            'series': {
                'productivity_score': series_score,
                'focus_time_minutes': series_focus,
                'tasks_completed': series_tasks
            },
            'summary': {
                'avg_productivity': round(np.mean([p['productivity_score'] for p in filtered]), 3),
                'total_focus_minutes': sum(p['focus_time_minutes'] for p in filtered),
                'total_tasks_completed': sum(p['tasks_completed'] for p in filtered)
            }
        }
    
    def get_task_analytics(self, period: str = 'week') -> Dict:
        """
        Get task completion analytics.
        
        Args:
            period: Time period
            
        Returns:
            Task analytics data
        """
        cutoff = datetime.now() - self.TIME_PERIODS.get(period, timedelta(days=7))
        
        filtered = [
            t for t in self.task_data
            if datetime.fromisoformat(t['timestamp']) > cutoff
        ]
        
        if not filtered:
            return {
                'success': False,
                'error': 'No task data available',
                'period': period
            }
        
        # Category breakdown
        by_category = defaultdict(list)
        for task in filtered:
            by_category[task.get('category', 'general')].append(task)
        
        category_stats = {}
        for category, tasks in by_category.items():
            completed = [t for t in tasks if t.get('completed', True)]
            category_stats[category] = {
                'total': len(tasks),
                'completed': len(completed),
                'completion_rate': round(len(completed) / len(tasks), 3) if tasks else 0
            }
        
        # Priority breakdown
        by_priority = defaultdict(list)
        for task in filtered:
            by_priority[task.get('priority', 'medium')].append(task)
        
        priority_stats = {}
        for priority, tasks in by_priority.items():
            completed = [t for t in tasks if t.get('completed', True)]
            priority_stats[priority] = {
                'total': len(tasks),
                'completed': len(completed),
                'completion_rate': round(len(completed) / len(tasks), 3) if tasks else 0
            }
        
        # Duration accuracy
        with_duration = [
            t for t in filtered 
            if t.get('estimated_duration') and t.get('actual_duration')
        ]
        
        if with_duration:
            estimation_accuracy = []
            for t in with_duration:
                actual = t.get('actual_duration')
                estimated = t.get('estimated_duration')
                if actual is not None and estimated is not None:
                    accuracy = 1 - abs(actual - estimated) / max(estimated, 1)
                    estimation_accuracy.append(max(0, accuracy))
            avg_accuracy = round(np.mean(estimation_accuracy), 3)
        else:
            avg_accuracy = None
        
        # Time of day analysis
        hour_completion = defaultdict(lambda: {'total': 0, 'completed': 0})
        for task in filtered:
            hour = datetime.fromisoformat(task['timestamp']).hour
            hour_completion[hour]['total'] += 1
            if task.get('completed', True):
                hour_completion[hour]['completed'] += 1
        
        best_hours = sorted(
            hour_completion.items(),
            key=lambda x: x[1]['completed'] / max(x[1]['total'], 1),
            reverse=True
        )[:5]
        
        return {
            'success': True,
            'period': period,
            'total_tasks': len(filtered),
            'completed_tasks': len([t for t in filtered if t.get('completed', True)]),
            'overall_completion_rate': round(
                len([t for t in filtered if t.get('completed', True)]) / len(filtered), 3
            ),
            'by_category': category_stats,
            'by_priority': priority_stats,
            'estimation_accuracy': avg_accuracy,
            'best_hours_for_completion': [
                {'hour': h, 'rate': round(d['completed'] / max(d['total'], 1), 2)}
                for h, d in best_hours
            ],
            'charts': {
                'category_pie': [
                    {'name': cat, 'value': stats['total']}
                    for cat, stats in category_stats.items()
                ],
                'priority_bar': [
                    {'name': pri, 'completed': stats['completed'], 'pending': stats['total'] - stats['completed']}
                    for pri, stats in priority_stats.items()
                ]
            }
        }
    
    def get_habit_analytics(self, period: str = 'month') -> Dict:
        """
        Get habit tracking analytics.
        
        Args:
            period: Time period
            
        Returns:
            Habit analytics data
        """
        cutoff = datetime.now() - self.TIME_PERIODS.get(period, timedelta(days=30))
        
        filtered = [
            h for h in self.habit_data
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
        
        if not filtered:
            return {
                'success': False,
                'error': 'No habit data available',
                'period': period
            }
        
        # Group by habit
        by_habit = defaultdict(list)
        for entry in filtered:
            habit_key = entry.get('habit_id') or entry.get('habit_name', 'unknown')
            by_habit[habit_key].append(entry)
        
        habit_stats = {}
        for habit_id, entries in by_habit.items():
            completed = [e for e in entries if e.get('completed', True)]
            streaks = [e.get('streak', 0) for e in entries]
            
            habit_stats[habit_id] = {
                'name': entries[0].get('habit_name', habit_id),
                'total_entries': len(entries),
                'completions': len(completed),
                'completion_rate': round(len(completed) / len(entries), 3) if entries else 0,
                'current_streak': entries[-1].get('streak', 0) if entries else 0,
                'max_streak': max(streaks) if streaks else 0,
                'avg_difficulty': round(np.mean([e.get('difficulty', 0.5) for e in entries]), 2)
            }
        
        # Overall metrics
        all_completions = [e for e in filtered if e.get('completed', True)]
        
        # Completion by day of week
        by_day = defaultdict(lambda: {'total': 0, 'completed': 0})
        for entry in filtered:
            day = datetime.fromisoformat(entry['timestamp']).weekday()
            by_day[day]['total'] += 1
            if entry.get('completed', True):
                by_day[day]['completed'] += 1
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_chart = [
            {
                'day': day_names[d],
                'completion_rate': round(data['completed'] / max(data['total'], 1), 2)
            }
            for d, data in sorted(by_day.items())
        ]
        
        return {
            'success': True,
            'period': period,
            'total_habits_tracked': len(by_habit),
            'total_entries': len(filtered),
            'overall_completion_rate': round(len(all_completions) / len(filtered), 3) if filtered else 0,
            'habits': habit_stats,
            'charts': {
                'completion_by_day': day_chart,
                'habit_comparison': [
                    {'name': stats['name'], 'completion_rate': stats['completion_rate']}
                    for habit_id, stats in habit_stats.items()
                ]
            }
        }
    
    def get_emotion_productivity_correlation(self, period: str = 'month') -> Dict:
        """
        Analyze correlation between emotions and productivity.
        
        Args:
            period: Time period
            
        Returns:
            Correlation analysis data
        """
        cutoff = datetime.now() - self.TIME_PERIODS.get(period, timedelta(days=30))
        
        # Get data from same time periods
        emotions = [
            e for e in self.emotion_data
            if datetime.fromisoformat(e['timestamp']) > cutoff
        ]
        productivity = [
            p for p in self.productivity_data
            if datetime.fromisoformat(p['timestamp']) > cutoff
        ]
        
        if len(emotions) < 10 or len(productivity) < 10:
            return {
                'success': False,
                'error': 'Insufficient data for correlation analysis',
                'period': period
            }
        
        # Group both by date
        emotion_by_date = defaultdict(list)
        for e in emotions:
            date = e['timestamp'][:10]
            emotion_by_date[date].append(e)
        
        productivity_by_date = defaultdict(list)
        for p in productivity:
            date = p['timestamp'][:10]
            productivity_by_date[date].append(p)
        
        # Calculate daily averages
        correlation_data = []
        for date in set(emotion_by_date.keys()) & set(productivity_by_date.keys()):
            avg_valence = np.mean([e['valence'] for e in emotion_by_date[date]])
            avg_arousal = np.mean([e['arousal'] for e in emotion_by_date[date]])
            avg_productivity = np.mean([p['productivity_score'] for p in productivity_by_date[date]])
            
            correlation_data.append({
                'date': date,
                'valence': avg_valence,
                'arousal': avg_arousal,
                'productivity': avg_productivity
            })
        
        if len(correlation_data) < 5:
            return {
                'success': False,
                'error': 'Insufficient overlapping data',
                'period': period
            }
        
        # Calculate correlations
        valences = [d['valence'] for d in correlation_data]
        arousals = [d['arousal'] for d in correlation_data]
        productivities = [d['productivity'] for d in correlation_data]
        
        valence_productivity_corr = np.corrcoef(valences, productivities)[0, 1]
        arousal_productivity_corr = np.corrcoef(arousals, productivities)[0, 1]
        
        # Insights
        insights = []
        if valence_productivity_corr > 0.5:
            insights.append({
                'finding': 'Strong positive correlation between mood and productivity',
                'implication': 'Better mood tends to lead to higher productivity',
                'suggestion': 'Prioritize activities that improve your mood'
            })
        elif valence_productivity_corr < -0.3:
            insights.append({
                'finding': 'Negative correlation between mood and productivity',
                'implication': 'This is unusual - consider reviewing the data',
                'suggestion': 'Explore what might be causing this pattern'
            })
        
        if arousal_productivity_corr > 0.4:
            insights.append({
                'finding': 'Higher energy levels correlate with productivity',
                'implication': 'Working during high-energy periods is beneficial',
                'suggestion': 'Schedule important tasks during high-energy times'
            })
        
        return {
            'success': True,
            'period': period,
            'data_points': len(correlation_data),
            'correlations': {
                'valence_productivity': round(valence_productivity_corr, 3),
                'arousal_productivity': round(arousal_productivity_corr, 3)
            },
            'interpretation': {
                'valence_productivity': self._interpret_correlation(valence_productivity_corr),
                'arousal_productivity': self._interpret_correlation(arousal_productivity_corr)
            },
            'insights': insights,
            'scatter_data': correlation_data
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        if np.isnan(r):
            return 'insufficient data'
        if r >= 0.7:
            return 'strong positive'
        elif r >= 0.4:
            return 'moderate positive'
        elif r >= 0.2:
            return 'weak positive'
        elif r >= -0.2:
            return 'negligible'
        elif r >= -0.4:
            return 'weak negative'
        elif r >= -0.7:
            return 'moderate negative'
        else:
            return 'strong negative'
    
    def get_dashboard_summary(self) -> Dict:
        """
        Get comprehensive dashboard summary with key metrics.
        
        Returns:
            Dashboard summary data
        """
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        week_start = today_start - timedelta(days=7)
        
        # Today's metrics
        today_emotions = [
            e for e in self.emotion_data
            if datetime.fromisoformat(e['timestamp']) > today_start
        ]
        today_productivity = [
            p for p in self.productivity_data
            if datetime.fromisoformat(p['timestamp']) > today_start
        ]
        today_tasks = [
            t for t in self.task_data
            if datetime.fromisoformat(t['timestamp']) > today_start
        ]
        
        # This week's metrics
        week_emotions = [
            e for e in self.emotion_data
            if datetime.fromisoformat(e['timestamp']) > week_start
        ]
        week_tasks = [
            t for t in self.task_data
            if datetime.fromisoformat(t['timestamp']) > week_start
        ]
        
        # Calculate key metrics
        current_mood = 'neutral'
        if today_emotions:
            current_mood = today_emotions[-1].get('emotion', 'neutral')
            avg_valence_today = np.mean([e['valence'] for e in today_emotions])
        else:
            avg_valence_today = 0
        
        if today_productivity:
            avg_productivity_today = np.mean([p['productivity_score'] for p in today_productivity])
            focus_time_today = sum(p['focus_time_minutes'] for p in today_productivity)
        else:
            avg_productivity_today = 0
            focus_time_today = 0
        
        tasks_completed_today = len([t for t in today_tasks if t.get('completed', True)])
        tasks_completed_week = len([t for t in week_tasks if t.get('completed', True)])
        
        # Trends
        if len(week_emotions) > 10:
            week_valences = [e['valence'] for e in week_emotions]
            mid = len(week_valences) // 2
            first_half_avg = np.mean(week_valences[:mid])
            second_half_avg = np.mean(week_valences[mid:])
            mood_trend = 'improving' if second_half_avg > first_half_avg + 0.1 else \
                        ('declining' if second_half_avg < first_half_avg - 0.1 else 'stable')
        else:
            mood_trend = 'insufficient data'
        
        return {
            'generated_at': now.isoformat(),
            'today': {
                'current_mood': current_mood,
                'avg_mood_valence': round(avg_valence_today, 2),
                'productivity_score': round(avg_productivity_today, 2),
                'focus_time_minutes': focus_time_today,
                'tasks_completed': tasks_completed_today
            },
            'this_week': {
                'emotion_entries': len(week_emotions),
                'tasks_completed': tasks_completed_week,
                'mood_trend': mood_trend
            },
            'quick_stats': {
                'total_emotion_entries': len(self.emotion_data),
                'total_tasks_tracked': len(self.task_data),
                'total_habits_tracked': len(set(h.get('habit_id') for h in self.habit_data))
            },
            'cards': [
                {
                    'title': 'Current Mood',
                    'value': current_mood.capitalize(),
                    'trend': mood_trend,
                    'icon': 'mood'
                },
                {
                    'title': "Today's Productivity",
                    'value': f"{round(avg_productivity_today * 100)}%",
                    'icon': 'productivity'
                },
                {
                    'title': 'Tasks Today',
                    'value': str(tasks_completed_today),
                    'icon': 'tasks'
                },
                {
                    'title': 'Focus Time',
                    'value': f"{focus_time_today}m",
                    'icon': 'focus'
                }
            ]
        }
    
    def _group_by_time(self, data: List[Dict], granularity: str) -> Dict[str, List]:
        """Group data by time granularity"""
        grouped = defaultdict(list)
        
        for entry in data:
            ts = datetime.fromisoformat(entry['timestamp'])
            
            if granularity == 'hour':
                key = ts.strftime('%Y-%m-%d %H:00')
            elif granularity == 'day':
                key = ts.strftime('%Y-%m-%d')
            elif granularity == 'week':
                # Get start of week
                start = ts - timedelta(days=ts.weekday())
                key = start.strftime('%Y-%m-%d')
            else:
                key = ts.strftime('%Y-%m-%d')
            
            grouped[key].append(entry)
        
        return dict(grouped)
    
    def export_analytics_report(self, 
                               period: str = 'month',
                               format: str = 'json') -> Dict:
        """
        Export comprehensive analytics report.
        
        Args:
            period: Time period for the report
            format: Export format ('json', 'summary')
            
        Returns:
            Complete analytics report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'period': period,
            'user_id': self.user_id,
            'sections': {}
        }
        
        # Add all analytics sections
        report['sections']['dashboard_summary'] = self.get_dashboard_summary()
        report['sections']['emotion_trends'] = self.get_emotion_trends(period, 'day')
        report['sections']['productivity_trends'] = self.get_productivity_trends(period, 'day')
        report['sections']['task_analytics'] = self.get_task_analytics(period)
        report['sections']['habit_analytics'] = self.get_habit_analytics(period)
        report['sections']['correlation_analysis'] = self.get_emotion_productivity_correlation(period)
        
        if format == 'summary':
            # Return condensed summary
            return {
                'generated_at': report['generated_at'],
                'period': period,
                'key_metrics': {
                    'avg_mood': report['sections']['emotion_trends'].get('summary', {}).get('avg_valence'),
                    'avg_productivity': report['sections']['productivity_trends'].get('summary', {}).get('avg_productivity'),
                    'task_completion_rate': report['sections']['task_analytics'].get('overall_completion_rate'),
                    'habit_completion_rate': report['sections']['habit_analytics'].get('overall_completion_rate')
                }
            }
        
        return report
