from django.contrib import admin
from .models import Task, Reminder, TaskCompletion, PredictiveInsight


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'category', 'priority', 'status', 'due_date', 'ai_suggested']
    list_filter = ['category', 'priority', 'status', 'ai_suggested']
    search_fields = ['title', 'description', 'user__username']
    readonly_fields = ['created_at', 'updated_at', 'completed_at']


@admin.register(Reminder)
class ReminderAdmin(admin.ModelAdmin):
    list_display = ['task', 'user', 'reminder_time', 'reminder_type', 'status']
    list_filter = ['reminder_type', 'status']
    search_fields = ['task__title', 'user__username']
    readonly_fields = ['created_at', 'sent_at']


@admin.register(TaskCompletion)
class TaskCompletionAdmin(admin.ModelAdmin):
    list_display = ['task', 'user', 'completed_at', 'completion_time', 'on_time', 'user_satisfaction']
    list_filter = ['on_time']
    search_fields = ['task__title', 'user__username']
    readonly_fields = ['completed_at']


@admin.register(PredictiveInsight)
class PredictiveInsightAdmin(admin.ModelAdmin):
    list_display = ['user', 'insight_type', 'title', 'confidence_score', 'is_acted_upon', 'created_at']
    list_filter = ['insight_type', 'is_acted_upon', 'user_feedback']
    search_fields = ['title', 'description', 'user__username']
    readonly_fields = ['created_at']
