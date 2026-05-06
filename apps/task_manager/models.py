from django.db import models
from django.conf import settings
from django.utils import timezone
from datetime import timedelta


class Task(models.Model):
    """User tasks and to-dos"""
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('overdue', 'Overdue'),
    ]
    
    CATEGORY_CHOICES = [
        ('work', 'Work'),
        ('personal', 'Personal'),
        ('health', 'Health'),
        ('shopping', 'Shopping'),
        ('social', 'Social'),
        ('learning', 'Learning'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='tasks')
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='other')
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='pending')
    
    # Scheduling
    due_date = models.DateTimeField(blank=True, null=True)
    estimated_duration = models.IntegerField(blank=True, null=True)  # in minutes
    
    # AI-suggested fields
    ai_suggested = models.BooleanField(default=False)
    ai_confidence = models.FloatField(default=0.0)  # 0-1
    
    # Completion tracking
    completed_at = models.DateTimeField(blank=True, null=True)
    actual_duration = models.IntegerField(blank=True, null=True)  # in minutes
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Recurrence
    is_recurring = models.BooleanField(default=False)
    recurrence_pattern = models.CharField(max_length=50, blank=True, null=True)  # daily, weekly, monthly
    
    def __str__(self):
        return f"{self.title} ({self.status})"
    
    def is_overdue(self):
        if self.due_date and self.status not in ['completed', 'cancelled']:
            return timezone.now() > self.due_date
        return False
    
    class Meta:
        ordering = ['-priority', 'due_date']


class Reminder(models.Model):
    """Reminders for tasks"""
    REMINDER_TYPE_CHOICES = [
        ('notification', 'Notification'),
        ('email', 'Email'),
        ('sms', 'SMS'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('dismissed', 'Dismissed'),
        ('snoozed', 'Snoozed'),
    ]
    
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='reminders')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='reminders')
    
    reminder_time = models.DateTimeField()
    reminder_type = models.CharField(max_length=15, choices=REMINDER_TYPE_CHOICES, default='notification')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    
    message = models.TextField(blank=True, null=True)
    
    # Snooze functionality
    snooze_until = models.DateTimeField(blank=True, null=True)
    snooze_count = models.IntegerField(default=0)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    sent_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"Reminder for {self.task.title} at {self.reminder_time}"
    
    class Meta:
        ordering = ['reminder_time']


class TaskCompletion(models.Model):
    """Task completion history for analytics"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='task_completions')
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='completion_history')
    
    completed_at = models.DateTimeField(auto_now_add=True)
    completion_time = models.IntegerField()  # actual time taken in minutes
    
    # Quality metrics
    on_time = models.BooleanField(default=True)
    user_satisfaction = models.IntegerField(blank=True, null=True)  # 1-5 rating
    
    # Context
    completion_notes = models.TextField(blank=True, null=True)
    emotional_state = models.CharField(max_length=20, blank=True, null=True)
    
    def __str__(self):
        return f"{self.task.title} completed on {self.completed_at}"
    
    class Meta:
        ordering = ['-completed_at']


class PredictiveInsight(models.Model):
    """ML-generated task predictions and suggestions"""
    INSIGHT_TYPE_CHOICES = [
        ('task_suggestion', 'Task Suggestion'),
        ('time_prediction', 'Time Prediction'),
        ('priority_suggestion', 'Priority Suggestion'),
        ('deadline_warning', 'Deadline Warning'),
        ('routine_detection', 'Routine Detection'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='insights')
    insight_type = models.CharField(max_length=30, choices=INSIGHT_TYPE_CHOICES)
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    
    # ML metadata
    confidence_score = models.FloatField()  # 0-1
    model_version = models.CharField(max_length=50, blank=True, null=True)
    
    # Action
    suggested_task = models.ForeignKey(Task, on_delete=models.CASCADE, blank=True, null=True, related_name='insights')
    is_acted_upon = models.BooleanField(default=False)
    user_feedback = models.CharField(max_length=20, blank=True, null=True)  # accepted, rejected, ignored
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.insight_type}: {self.title}"
    
    class Meta:
        ordering = ['-confidence_score', '-created_at']
