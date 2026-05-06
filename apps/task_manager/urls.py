"""
Task Manager URLs
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.task_list_view, name='task_list'),
    path('create/', views.create_task_view, name='create_task'),
    path('<int:task_id>/complete/', views.complete_task, name='complete_task'),
    path('<int:task_id>/delete/', views.delete_task, name='delete_task'),
    path('reminder-feed/', views.reminder_feed, name='reminder_feed'),
]
