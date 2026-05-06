from django.urls import path

from . import views

app_name = 'planner'

urlpatterns = [
    path('', views.planner_home, name='home'),
]
