"""
Celery configuration for Personal AI Assistant
"""
import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('personal_assistant')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Periodic tasks configuration
app.conf.beat_schedule = {
    'check-pending-reminders': {
        'task': 'apps.task_manager.tasks.check_pending_reminders',
        'schedule': 60.0,  # Every minute
    },
    'analyze-user-behavior': {
        'task': 'apps.emotion_tracker.tasks.analyze_daily_patterns',
        'schedule': crontab(hour=0, minute=0),  # Daily at midnight
    },
}

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
