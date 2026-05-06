from django.urls import path

from . import views

app_name = 'voice_assistant'

urlpatterns = [
    path('', views.voice_assistant_home, name='home'),
    path('language-bridge/', views.language_bridge, name='language_bridge'),
    path('command/', views.process_command, name='process_command'),
    path('transcribe/', views.transcribe_voice, name='transcribe_voice'),
]
