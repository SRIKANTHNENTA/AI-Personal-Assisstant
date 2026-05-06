"""Project websocket routing."""

from apps.chat_companion.routing import websocket_urlpatterns as chat_websocket_urlpatterns
from apps.admin_dashboard.routing import websocket_urlpatterns as dashboard_websocket_urlpatterns
from apps.music.routing import websocket_urlpatterns as music_websocket_urlpatterns

websocket_urlpatterns = chat_websocket_urlpatterns + dashboard_websocket_urlpatterns + music_websocket_urlpatterns