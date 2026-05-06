from django.db import models
from django.conf import settings
from django.utils import timezone


# Admin dashboard uses models from other apps for analytics
# This file contains admin-specific models for system monitoring

class SystemLog(models.Model):
    """System-wide logs for admin monitoring"""
    LOG_LEVEL_CHOICES = [
        ('debug', 'Debug'),
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('critical', 'Critical'),
    ]
    
    log_level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES)
    module = models.CharField(max_length=100)
    message = models.TextField()
    
    # Additional context
    stack_trace = models.TextField(blank=True, null=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Metadata
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    
    def __str__(self):
        return f"[{self.log_level.upper()}] {self.module}: {self.message[:50]}"
    
    class Meta:
        ordering = ['-timestamp']


class APIUsageLog(models.Model):
    """Track external API usage for monitoring costs"""
    API_TYPE_CHOICES = [
        ('openai', 'OpenAI'),
        ('google_speech', 'Google Speech'),
        ('google_translate', 'Google Translate'),
        ('azure_emotion', 'Azure Emotion'),
        ('other', 'Other'),
    ]
    
    api_type = models.CharField(max_length=20, choices=API_TYPE_CHOICES)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='api_usage')
    
    # Request details
    endpoint = models.CharField(max_length=200)
    request_data = models.JSONField(blank=True, null=True)
    response_data = models.JSONField(blank=True, null=True)
    
    # Metrics
    tokens_used = models.IntegerField(blank=True, null=True)
    response_time = models.FloatField(blank=True, null=True)  # in seconds
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True, null=True)
    
    # Cost tracking
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=6, blank=True, null=True)
    
    # Timestamps
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.api_type}: {self.endpoint} ({self.timestamp})"
    
    class Meta:
        ordering = ['-timestamp']


class UserStatistics(models.Model):
    """Daily user statistics for admin dashboard"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='statistics')
    date = models.DateField()
    
    # Activity metrics
    total_messages = models.IntegerField(default=0)
    total_tasks_created = models.IntegerField(default=0)
    total_tasks_completed = models.IntegerField(default=0)
    total_voice_inputs = models.IntegerField(default=0)
    total_camera_uses = models.IntegerField(default=0)
    
    # Time metrics
    total_active_time = models.IntegerField(default=0)  # in seconds
    average_session_duration = models.IntegerField(default=0)  # in seconds
    
    # Engagement metrics
    conversation_count = models.IntegerField(default=0)
    average_sentiment = models.FloatField(default=0.0)  # -1 to 1
    dominant_emotion = models.CharField(max_length=20, blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.date}"
    
    class Meta:
        ordering = ['-date']
        unique_together = ['user', 'date']
