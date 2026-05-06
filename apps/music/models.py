from django.conf import settings
from django.db import models


class Track(models.Model):
	SOURCE_CHOICES = [
		("local", "Local Library"),
		("spotify", "Spotify"),
	]

	track_id = models.CharField(max_length=120, unique=True)
	title = models.CharField(max_length=200)
	artist = models.CharField(max_length=200, blank=True)
	source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default="local")
	preview_url = models.URLField(blank=True)
	stream_url = models.URLField(blank=True)
	image_url = models.URLField(blank=True)

	# Audio metadata for mood scoring (range 0.0 - 1.0)
	valence = models.FloatField(default=0.5)
	energy = models.FloatField(default=0.5)

	duration_seconds = models.PositiveIntegerField(default=0)
	is_active = models.BooleanField(default=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self):
		return f"{self.title} - {self.artist or 'Unknown'}"

	class Meta:
		ordering = ["title"]


class MusicSession(models.Model):
	PLAYBACK_STATES = [
		("playing", "Playing"),
		("paused", "Paused"),
		("stopped", "Stopped"),
	]

	user = models.ForeignKey(
		settings.AUTH_USER_MODEL,
		on_delete=models.CASCADE,
		related_name="music_sessions",
	)
	session_key = models.CharField(max_length=80)
	queue_track_ids = models.JSONField(default=list)
	current_index = models.PositiveIntegerField(default=0)
	current_track = models.ForeignKey(
		Track,
		null=True,
		blank=True,
		on_delete=models.SET_NULL,
		related_name="active_sessions",
	)
	playback_state = models.CharField(max_length=20, choices=PLAYBACK_STATES, default="paused")
	current_time_seconds = models.FloatField(default=0.0)
	volume = models.FloatField(default=0.8)

	last_mood = models.CharField(max_length=30, blank=True)
	last_mood_shift_at = models.DateTimeField(null=True, blank=True)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self):
		return f"MusicSession(user={self.user_id}, session={self.session_key})"

	class Meta:
		unique_together = ("user", "session_key")
		ordering = ["-updated_at"]
