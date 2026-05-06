"""WebSocket consumer for live dashboard updates."""

from __future__ import annotations

import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth import get_user_model

from apps.chat_companion.models import Conversation
from apps.emotion_tracker.models import EmotionState
from apps.task_manager.models import Task
from ai_services.cognitive_ai_integration import CognitiveAIServices

User = get_user_model()


class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        self.group_name = f"dashboard_{self.user.id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        await self.send_state()

    async def disconnect(self, close_code):
        if getattr(self, "group_name", None):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data or "{}")
        if data.get("type") in {"ping", "request_state"}:
            await self.send_state()

    async def send_state(self):
        state = await self.get_state()
        await self.send(text_data=json.dumps({"type": "dashboard_state", "state": state}))

    @sync_to_async
    def get_state(self):
        services = CognitiveAIServices(str(self.user.id))
        cognitive_state = services.get_cognitive_state()
        recent_emotion = EmotionState.objects.filter(user=self.user).order_by("-detected_at").first()
        pending_tasks = Task.objects.filter(user=self.user, status="pending").count()
        recent_conversations = Conversation.objects.filter(user=self.user).count()

        return {
            "status": "online",
            "pending_tasks": pending_tasks,
            "recent_conversations": recent_conversations,
            "focus_level": cognitive_state.get("focus_level", 0.0),
            "energy_level": cognitive_state.get("energy_level", 0.0),
            "stress_level": cognitive_state.get("stress_level", 0.0),
            "recent_emotion": getattr(recent_emotion, "emotion", "steady"),
        }