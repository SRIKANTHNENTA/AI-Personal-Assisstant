"""
Cognitive AI API URL Configuration
Routes for all cognitive AI service endpoints.
"""

from django.urls import path
from .cognitive_ai_views import (
    EmotionAnalysisView,
    CognitiveStateView,
    ActivityRecordView,
    BehaviorPatternsView,
    TaskRecommendationsView,
    TaskCompletionView,
    HabitCreateView,
    HabitLogView,
    DailyHabitsView,
    ConversationProcessView,
    ResponseStyleView,
    SystemPromptView,
    DashboardView,
    EmotionTrendsView,
    ProductivityTrendsView,
    AnalyticsReportView,
    CorrelationAnalysisView,
    RiskAssessmentView,
    AnomalySummaryView,
    WellnessInsightsView,
    ExplainRecommendationView,
    TrainModelsView,
    ServiceStatusView,
)

app_name = 'cognitive_ai'

urlpatterns = [
    # Emotion Analysis
    path('emotion/analyze/', EmotionAnalysisView.as_view(), name='emotion_analyze'),
    path('cognitive-state/', CognitiveStateView.as_view(), name='cognitive_state'),
    
    # Behavior Modeling
    path('activity/record/', ActivityRecordView.as_view(), name='activity_record'),
    path('behavior/patterns/', BehaviorPatternsView.as_view(), name='behavior_patterns'),
    
    # Task Scheduling
    path('tasks/recommend/', TaskRecommendationsView.as_view(), name='task_recommend'),
    path('tasks/complete/', TaskCompletionView.as_view(), name='task_complete'),
    
    # Habit Tracking
    path('habits/create/', HabitCreateView.as_view(), name='habit_create'),
    path('habits/log/', HabitLogView.as_view(), name='habit_log'),
    path('habits/today/', DailyHabitsView.as_view(), name='habits_today'),
    
    # Personality & Conversation
    path('conversation/process/', ConversationProcessView.as_view(), name='conversation_process'),
    path('personality/style/', ResponseStyleView.as_view(), name='personality_style'),
    path('personality/prompt/', SystemPromptView.as_view(), name='system_prompt'),
    
    # Analytics
    path('analytics/dashboard/', DashboardView.as_view(), name='analytics_dashboard'),
    path('analytics/emotions/', EmotionTrendsView.as_view(), name='emotion_trends'),
    path('analytics/productivity/', ProductivityTrendsView.as_view(), name='productivity_trends'),
    path('analytics/report/', AnalyticsReportView.as_view(), name='analytics_report'),
    path('analytics/correlation/', CorrelationAnalysisView.as_view(), name='correlation_analysis'),
    
    # Anomaly Detection & Wellness
    path('anomaly/risk/', RiskAssessmentView.as_view(), name='risk_assessment'),
    path('anomaly/summary/', AnomalySummaryView.as_view(), name='anomaly_summary'),
    path('wellness/insights/', WellnessInsightsView.as_view(), name='wellness_insights'),
    
    # Explainability
    path('explain/', ExplainRecommendationView.as_view(), name='explain'),
    
    # Model Management
    path('models/train/', TrainModelsView.as_view(), name='train_models'),
    path('status/', ServiceStatusView.as_view(), name='service_status'),
]
