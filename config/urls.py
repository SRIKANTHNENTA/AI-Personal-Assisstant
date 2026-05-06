"""
URL configuration for Personal AI Assistant project.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from apps.task_manager import views as task_views
from django.views.generic import TemplateView
from django.views.generic.base import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Home
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    
    # Dashboard
    path('dashboard/', task_views.dashboard_view, name='dashboard'),
    
    # App URLs
    path('auth/', include('apps.authentication.urls')),
    path('chat/', include('apps.chat_companion.urls')),
    path('tasks/', include('apps.task_manager.urls')),
    path('voice/', include('apps.voice_assistant.urls')),
    path('notes/', include('apps.notes.urls')),
    path('vision/', include('apps.vision.urls')),
    path('music/', include('apps.music.urls')),
    path('finance/', RedirectView.as_view(pattern_name='dashboard', permanent=False)),
    path('email/', include('apps.email_ai.urls')),
    
    # Cognitive AI API
    path('api/ai/', include('api.urls', namespace='cognitive_ai')),
]

# Media files
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

