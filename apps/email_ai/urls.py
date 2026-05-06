from django.urls import path

from . import views

app_name = 'email_ai'

urlpatterns = [
    path('', views.email_home, name='home'),
]
