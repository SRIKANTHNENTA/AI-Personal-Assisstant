from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class User(AbstractUser):
    """Extended User model with additional fields"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('admin', 'Admin'),
    ]
    
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.username


class UserPreferences(models.Model):
    """User preferences and settings"""
    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('es', 'Spanish'),
        ('fr', 'French'),
        ('de', 'German'),
        ('hi', 'Hindi'),
        ('zh', 'Chinese'),
    ]
    
    THEME_CHOICES = [
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('auto', 'Auto'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='preferences')
    preferred_language = models.CharField(max_length=5, choices=LANGUAGE_CHOICES, default='en')
    theme = models.CharField(max_length=10, choices=THEME_CHOICES, default='auto')
    enable_voice_input = models.BooleanField(default=True)
    enable_camera_emotion = models.BooleanField(default=True)
    enable_notifications = models.BooleanField(default=True)
    enable_email_notifications = models.BooleanField(default=False)
    notification_sound = models.BooleanField(default=True)
    timezone = models.CharField(max_length=50, default='Asia/Kolkata')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Preferences"
    
    class Meta:
        verbose_name_plural = "User Preferences"


class UserBehaviorProfile(models.Model):
    """ML-generated user behavior insights"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='behavior_profile')
    
    # Routine patterns (JSON field storing detected patterns)
    morning_routine_time = models.TimeField(blank=True, null=True)
    evening_routine_time = models.TimeField(blank=True, null=True)
    most_active_hours = models.JSONField(default=list, blank=True)
    
    # Task patterns
    average_task_completion_time = models.FloatField(default=0.0)  # in minutes
    preferred_task_categories = models.JSONField(default=list, blank=True)
    task_completion_rate = models.FloatField(default=0.0)  # percentage
    
    # Emotional patterns
    dominant_emotion = models.CharField(max_length=20, blank=True, null=True)
    emotional_stability_score = models.FloatField(default=0.0)  # 0-100
    
    # Interaction patterns
    average_conversation_length = models.IntegerField(default=0)  # messages per conversation
    preferred_communication_style = models.CharField(max_length=50, blank=True, null=True)
    
    # ML model metadata
    last_analysis_date = models.DateTimeField(auto_now=True)
    analysis_confidence = models.FloatField(default=0.0)  # 0-1
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Behavior Profile"
