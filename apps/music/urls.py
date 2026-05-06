from django.urls import path

from . import views

app_name = 'music'

urlpatterns = [
    path('', views.music_home, name='home'),
    path('state/', views.music_state, name='state'),
    path('player/action/', views.player_action, name='player_action'),
    path('stream/<int:track_id>/', views.track_stream, name='stream'),
]
