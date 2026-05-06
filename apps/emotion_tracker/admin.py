from django.contrib import admin
from .models import (
    EmotionState, EmotionHistory, VoiceToneAnalysis,
    FacialEmotionLog, BehaviorPattern, UserActivity
)


@admin.register(EmotionState)
class EmotionStateAdmin(admin.ModelAdmin):
    list_display = ['user', 'emotion', 'intensity', 'source', 'confidence_score', 'detected_at']
    list_filter = ['emotion', 'source']
    search_fields = ['user__username']
    readonly_fields = ['detected_at']


@admin.register(EmotionHistory)
class EmotionHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'date', 'dominant_emotion', 'emotional_stability_score', 'stress_level']
    list_filter = ['dominant_emotion', 'date']
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(VoiceToneAnalysis)
class VoiceToneAnalysisAdmin(admin.ModelAdmin):
    list_display = ['user', 'detected_emotion', 'emotion_confidence', 'voice_stress_level', 'analyzed_at']
    list_filter = ['detected_emotion']
    search_fields = ['user__username']
    readonly_fields = ['analyzed_at']


@admin.register(FacialEmotionLog)
class FacialEmotionLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'primary_emotion', 'primary_confidence', 'smile_intensity', 'detected_at']
    list_filter = ['primary_emotion']
    search_fields = ['user__username']
    readonly_fields = ['detected_at']


@admin.register(BehaviorPattern)
class BehaviorPatternAdmin(admin.ModelAdmin):
    list_display = ['user', 'pattern_type', 'name', 'confidence_score', 'occurrence_count', 'is_active']
    list_filter = ['pattern_type', 'is_active']
    search_fields = ['user__username', 'name']
    readonly_fields = ['first_detected', 'last_updated']


@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ['user', 'activity_type', 'timestamp', 'emotional_state']
    list_filter = ['activity_type', 'emotional_state']
    search_fields = ['user__username', 'description']
    readonly_fields = ['timestamp']
