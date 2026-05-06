from django.db import models
from django.conf import settings
from django.utils import timezone


class EmotionState(models.Model):
    """Real-time emotion detection and tracking"""
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('fearful', 'Fearful'),
        ('surprised', 'Surprised'),
        ('disgusted', 'Disgusted'),
        ('neutral', 'Neutral'),
        ('excited', 'Excited'),
        ('anxious', 'Anxious'),
        ('calm', 'Calm'),
        ('stressed', 'Stressed'),
        ('tired', 'Tired'),
    ]
    
    SOURCE_CHOICES = [
        ('text', 'Text Analysis'),
        ('voice', 'Voice Tone'),
        ('facial', 'Facial Expression'),
        ('combined', 'Combined Sources'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='emotion_states')
    
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    intensity = models.FloatField()  # 0-1
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES)
    
    # Context
    detected_at = models.DateTimeField(auto_now_add=True)
    context_description = models.TextField(blank=True, null=True)
    
    # Source-specific data
    text_content = models.TextField(blank=True, null=True)
    voice_features = models.JSONField(blank=True, null=True)  # pitch, tone, etc.
    facial_features = models.JSONField(blank=True, null=True)  # facial landmarks
    
    # ML metadata
    confidence_score = models.FloatField(default=0.0)  # 0-1
    model_version = models.CharField(max_length=50, blank=True, null=True)
    
    def __str__(self):
        return f"{self.user.username}: {self.emotion} ({self.intensity:.2f}) at {self.detected_at}"
    
    class Meta:
        ordering = ['-detected_at']


class EmotionHistory(models.Model):
    """Aggregated emotional patterns over time"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='emotion_history')
    
    date = models.DateField()
    
    # Daily emotion distribution (percentage)
    happy_percentage = models.FloatField(default=0.0)
    sad_percentage = models.FloatField(default=0.0)
    angry_percentage = models.FloatField(default=0.0)
    anxious_percentage = models.FloatField(default=0.0)
    neutral_percentage = models.FloatField(default=0.0)
    other_percentage = models.FloatField(default=0.0)
    
    # Daily metrics
    dominant_emotion = models.CharField(max_length=20)
    average_intensity = models.FloatField()
    emotion_changes_count = models.IntegerField(default=0)
    
    # Stability metrics
    emotional_stability_score = models.FloatField(default=0.0)  # 0-100
    stress_level = models.FloatField(default=0.0)  # 0-100
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.date}: {self.dominant_emotion}"
    
    class Meta:
        ordering = ['-date']
        unique_together = ['user', 'date']


class VoiceToneAnalysis(models.Model):
    """Voice-based emotion and tone analysis"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='voice_analyses')
    
    audio_file = models.FileField(upload_to='voice_analysis/')
    duration = models.FloatField()  # in seconds
    
    # Voice features
    pitch_average = models.FloatField(blank=True, null=True)
    pitch_variance = models.FloatField(blank=True, null=True)
    speech_rate = models.FloatField(blank=True, null=True)  # words per minute
    volume_average = models.FloatField(blank=True, null=True)
    
    # Emotion detection
    detected_emotion = models.CharField(max_length=20)
    emotion_confidence = models.FloatField()
    
    # Stress indicators
    voice_stress_level = models.FloatField(default=0.0)  # 0-100
    hesitation_count = models.IntegerField(default=0)
    
    # Metadata
    analyzed_at = models.DateTimeField(auto_now_add=True)
    transcription = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"Voice Analysis: {self.detected_emotion} ({self.analyzed_at})"
    
    class Meta:
        ordering = ['-analyzed_at']


class FacialEmotionLog(models.Model):
    """Camera-based facial emotion detection"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='facial_emotions')
    
    # Image data
    image_file = models.ImageField(upload_to='facial_analysis/', blank=True, null=True)
    
    # Detected emotions (can have multiple with different intensities)
    primary_emotion = models.CharField(max_length=20)
    primary_confidence = models.FloatField()
    
    secondary_emotion = models.CharField(max_length=20, blank=True, null=True)
    secondary_confidence = models.FloatField(blank=True, null=True)
    
    # Facial features
    facial_landmarks = models.JSONField(blank=True, null=True)
    face_position = models.JSONField(blank=True, null=True)  # x, y, width, height
    
    # Additional metrics
    smile_intensity = models.FloatField(default=0.0)  # 0-1
    eye_contact = models.BooleanField(default=False)
    head_pose = models.JSONField(blank=True, null=True)  # pitch, yaw, roll
    
    # Metadata
    detected_at = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, blank=True, null=True)
    
    def __str__(self):
        return f"Facial: {self.primary_emotion} ({self.detected_at})"
    
    class Meta:
        ordering = ['-detected_at']


class BehaviorPattern(models.Model):
    """ML-detected behavioral patterns"""
    PATTERN_TYPE_CHOICES = [
        ('routine', 'Routine'),
        ('habit', 'Habit'),
        ('trend', 'Trend'),
        ('anomaly', 'Anomaly'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='behavior_patterns')
    
    pattern_type = models.CharField(max_length=20, choices=PATTERN_TYPE_CHOICES)
    name = models.CharField(max_length=200)
    description = models.TextField()
    
    # Pattern details
    frequency = models.CharField(max_length=50)  # daily, weekly, monthly
    time_of_day = models.TimeField(blank=True, null=True)
    day_of_week = models.IntegerField(blank=True, null=True)  # 0-6
    
    # ML metrics
    confidence_score = models.FloatField()  # 0-1
    occurrence_count = models.IntegerField(default=0)
    last_occurred = models.DateTimeField(blank=True, null=True)
    
    # Pattern data
    pattern_data = models.JSONField(blank=True, null=True)  # Additional pattern-specific data
    
    # Timestamps
    first_detected = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.user.username}: {self.name} ({self.pattern_type})"
    
    class Meta:
        ordering = ['-confidence_score', '-last_updated']


class UserActivity(models.Model):
    """Activity logs for big data analysis"""
    ACTIVITY_TYPE_CHOICES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('chat', 'Chat Message'),
        ('task_create', 'Task Created'),
        ('task_complete', 'Task Completed'),
        ('voice_input', 'Voice Input'),
        ('camera_access', 'Camera Access'),
        ('settings_change', 'Settings Change'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='activities')
    
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPE_CHOICES)
    description = models.TextField(blank=True, null=True)
    
    # Activity metadata
    activity_data = models.JSONField(blank=True, null=True)
    
    # Context
    emotional_state = models.CharField(max_length=20, blank=True, null=True)
    device_info = models.JSONField(blank=True, null=True)
    
    # Timestamps
    timestamp = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(blank=True, null=True)  # in seconds
    
    def __str__(self):
        return f"{self.user.username}: {self.activity_type} at {self.timestamp}"
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'activity_type', 'timestamp']),
        ]
