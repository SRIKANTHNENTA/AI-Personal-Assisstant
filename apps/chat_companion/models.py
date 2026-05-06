from django.db import models
from django.conf import settings
from django.utils import timezone


class Conversation(models.Model):
    """Conversation session tracking"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=200, blank=True, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    last_message_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Conversation metadata
    message_count = models.IntegerField(default=0)
    detected_language = models.CharField(max_length=10, default='en')
    
    def __str__(self):
        return f"Conversation {self.id} - {self.user.username}"
    
    class Meta:
        ordering = ['-last_message_at']


class Message(models.Model):
    """Individual messages in conversations"""
    MESSAGE_TYPE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    INPUT_MODE_CHOICES = [
        ('text', 'Text'),
        ('voice', 'Voice'),
        ('command', 'Command'),
    ]
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPE_CHOICES)
    content = models.TextField()
    input_mode = models.CharField(max_length=10, choices=INPUT_MODE_CHOICES, default='text')
    
    # Emotion and sentiment
    detected_emotion = models.CharField(max_length=20, blank=True, null=True)
    sentiment_score = models.FloatField(blank=True, null=True)  # -1 to 1
    
    # XAI and Cognitive Insights
    explanation = models.TextField(blank=True, null=True)  # SHAP/LIME explanation
    cognitive_impact = models.JSONField(default=dict, blank=True)  # Real-time impact on user's state
    
    # Metadata
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    # Voice-specific fields
    audio_file = models.FileField(upload_to='voice_messages/', blank=True, null=True)
    audio_duration = models.FloatField(blank=True, null=True)  # in seconds
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}"
    
    class Meta:
        ordering = ['timestamp']


class ConversationContext(models.Model):
    """Stores conversation context for AI understanding"""
    conversation = models.OneToOneField(Conversation, on_delete=models.CASCADE, related_name='context')
    
    # Context data
    current_topic = models.CharField(max_length=200, blank=True, null=True)
    mentioned_entities = models.JSONField(default=list, blank=True)  # Names, places, etc.
    user_intent = models.CharField(max_length=100, blank=True, null=True)
    
    # Conversation state
    pending_tasks = models.JSONField(default=list, blank=True)
    discussed_topics = models.JSONField(default=list, blank=True)
    
    # Memory references
    referenced_past_conversations = models.JSONField(default=list, blank=True)
    
    # Metadata
    last_updated = models.DateTimeField(auto_now=True)
    context_summary = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"Context for Conversation {self.conversation.id}"


class UserMemory(models.Model):
    """Long-term memory storage for personalization"""
    MEMORY_TYPE_CHOICES = [
        ('preference', 'Preference'),
        ('fact', 'Fact'),
        ('event', 'Event'),
        ('relationship', 'Relationship'),
        ('habit', 'Habit'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='memories')
    memory_type = models.CharField(max_length=20, choices=MEMORY_TYPE_CHOICES)
    
    key = models.CharField(max_length=200)  # e.g., "favorite_color", "mother_name"
    value = models.TextField()
    
    # Confidence and importance
    confidence_score = models.FloatField(default=1.0)  # 0-1
    importance_score = models.FloatField(default=0.5)  # 0-1
    
    # Source tracking
    source_conversation = models.ForeignKey(Conversation, on_delete=models.SET_NULL, null=True, blank=True)
    learned_from_message = models.ForeignKey(Message, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    last_referenced = models.DateTimeField(auto_now=True)
    reference_count = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.user.username}: {self.key} = {self.value[:50]}"
    
    class Meta:
        ordering = ['-importance_score', '-last_referenced']
        unique_together = ['user', 'key']
