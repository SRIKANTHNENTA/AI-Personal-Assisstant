"""
Chat Companion URLs
"""

from django.urls import path
from . import views

app_name = 'chat_companion'

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('thread/<int:conversation_id>/', views.conversation_thread, name='conversation_thread'),
    path('send/', views.send_message, name='send_message'),
    path('process-voice/', views.process_voice, name='process_voice'),
    path('history/', views.conversation_history, name='conversation_history'),
    path('new/', views.new_conversation, name='new_conversation'),
]
