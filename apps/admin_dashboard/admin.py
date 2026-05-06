from django.contrib import admin
from .models import SystemLog, APIUsageLog, UserStatistics


@admin.register(SystemLog)
class SystemLogAdmin(admin.ModelAdmin):
    list_display = ['log_level', 'module', 'message', 'user', 'timestamp']
    list_filter = ['log_level', 'module']
    search_fields = ['message', 'user__username']
    readonly_fields = ['timestamp']


@admin.register(APIUsageLog)
class APIUsageLogAdmin(admin.ModelAdmin):
    list_display = ['api_type', 'user', 'endpoint', 'success', 'tokens_used', 'estimated_cost', 'timestamp']
    list_filter = ['api_type', 'success']
    search_fields = ['user__username', 'endpoint']
    readonly_fields = ['timestamp']


@admin.register(UserStatistics)
class UserStatisticsAdmin(admin.ModelAdmin):
    list_display = ['user', 'date', 'total_messages', 'total_tasks_completed', 'dominant_emotion']
    list_filter = ['date', 'dominant_emotion']
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'updated_at']
