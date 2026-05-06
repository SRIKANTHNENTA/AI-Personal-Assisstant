from __future__ import annotations

import asyncio
import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from apps.emotion_tracker.models import EmotionState
from apps.music.models import MusicSession
from apps.music.views import refresh_session_for_mood_shift


class MusicConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        self.group_name = f"music_{self.user.id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

        await self.send_state(reason="connect")
        self.polling_task = asyncio.create_task(self._poll_updates())

    async def disconnect(self, close_code):
        if getattr(self, "polling_task", None):
            self.polling_task.cancel()
        if getattr(self, "group_name", None):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data or "{}")
        if data.get("type") in {"ping", "request_state"}:
            await self.send_state(reason="manual")

    async def _poll_updates(self):
        try:
            while True:
                await self.send_state(reason="poll")
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            return

    async def send_state(self, reason: str):
        state = await self.get_live_state()
        await self.send(text_data=json.dumps({"type": "music_state", "reason": reason, "state": state}))

    @sync_to_async
    def get_live_state(self):
        scope_session = self.scope.get("session")
        session_key = getattr(scope_session, "session_key", None) or "ws-default"
        session, _ = MusicSession.objects.get_or_create(
            user=self.user,
            session_key=session_key,
            defaults={"volume": 0.8, "playback_state": "paused", "queue_track_ids": []},
        )

        recent_emotion = EmotionState.objects.filter(user=self.user).order_by("-detected_at").first()
        mood = getattr(recent_emotion, "emotion", "neutral")
        intensity = getattr(recent_emotion, "intensity", 0.5)

        return refresh_session_for_mood_shift(
            session=session,
            mood=mood,
            intensity=intensity,
            cooldown_seconds=45,
        )
