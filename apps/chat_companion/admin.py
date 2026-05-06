from django.contrib import admin
from .models import Conversation, Message, ConversationContext, UserMemory


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'title', 'message_count', 'is_active', 'last_message_at']
    list_filter = ['is_active', 'detected_language']
    search_fields = ['user__username', 'title']
    readonly_fields = ['started_at', 'last_message_at']


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['id', 'conversation', 'message_type', 'input_mode', 'detected_emotion', 'timestamp']
    list_filter = ['message_type', 'input_mode', 'detected_emotion']
    search_fields = ['content', 'conversation__user__username']
    readonly_fields = ['timestamp']


@admin.register(ConversationContext)
class ConversationContextAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'current_topic', 'user_intent', 'last_updated']
    search_fields = ['conversation__user__username', 'current_topic']
    readonly_fields = ['last_updated']


@admin.register(UserMemory)
class UserMemoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'memory_type', 'key', 'confidence_score', 'importance_score', 'reference_count']
    list_filter = ['memory_type']
    search_fields = ['user__username', 'key', 'value']
    readonly_fields = ['created_at', 'last_referenced']
