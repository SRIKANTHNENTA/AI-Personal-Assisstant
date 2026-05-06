from __future__ import annotations

import json
import io
import math
import struct
import wave
import time

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST
import requests

from apps.emotion_tracker.models import EmotionState
from apps.music.models import MusicSession, Track

MOOD_AUDIO_PROFILE = {
    "happy": {"valence": 0.88, "energy": 0.82},
    "excited": {"valence": 0.9, "energy": 0.92},
    "calm": {"valence": 0.58, "energy": 0.34},
    "neutral": {"valence": 0.5, "energy": 0.5},
    "sad": {"valence": 0.24, "energy": 0.22},
    "angry": {"valence": 0.3, "energy": 0.84},
    "anxious": {"valence": 0.36, "energy": 0.76},
    "stressed": {"valence": 0.28, "energy": 0.8},
    "tired": {"valence": 0.4, "energy": 0.2},
    "fearful": {"valence": 0.18, "energy": 0.72},
    "surprised": {"valence": 0.65, "energy": 0.72},
    "disgusted": {"valence": 0.16, "energy": 0.66},
}

DEFAULT_TRACKS = [
    {
        "track_id": "local_neon_drive_01",
        "title": "Neon Drive",
        "artist": "Future Grid",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "valence": 0.84,
        "energy": 0.88,
        "duration_seconds": 360,
    },
    {
        "track_id": "local_focus_current_02",
        "title": "Focus Current",
        "artist": "Node Theory",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "valence": 0.56,
        "energy": 0.48,
        "duration_seconds": 360,
    },
    {
        "track_id": "local_soft_orbit_03",
        "title": "Soft Orbit",
        "artist": "Luma Sleep",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "valence": 0.34,
        "energy": 0.26,
        "duration_seconds": 348,
    },
    {
        "track_id": "local_restore_04",
        "title": "Restore Protocol",
        "artist": "Calm Systems",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
        "valence": 0.44,
        "energy": 0.38,
        "duration_seconds": 350,
    },
    {
        "track_id": "local_heat_sink_05",
        "title": "Heat Sink",
        "artist": "Pulse Logic",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
        "valence": 0.28,
        "energy": 0.86,
        "duration_seconds": 360,
    },
    {
        "track_id": "local_bloom_06",
        "title": "Morning Bloom",
        "artist": "Aurora Lane",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
        "valence": 0.92,
        "energy": 0.7,
        "duration_seconds": 371,
    },
    {
        "track_id": "local_ocean_code_07",
        "title": "Ocean Code",
        "artist": "Quiet Stack",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
        "valence": 0.5,
        "energy": 0.3,
        "duration_seconds": 364,
    },
    {
        "track_id": "local_reframe_08",
        "title": "Reframe",
        "artist": "Signal Kind",
        "source": "local",
        "stream_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        "valence": 0.62,
        "energy": 0.54,
        "duration_seconds": 355,
    },
]

_INDUSTRY_RECO_CACHE: dict[tuple[str, str], dict[str, object]] = {}
_INDUSTRY_RECO_TTL_SECONDS = 1200

INDUSTRY_QUERY_CONFIG = {
    "bollywood": {"label": "Bollywood", "country": "IN", "base_terms": ["bollywood", "hindi soundtrack"]},
    "hollywood": {"label": "Hollywood", "country": "US", "base_terms": ["hollywood soundtrack", "movie theme"]},
    "tollywood": {"label": "Tollywood", "country": "IN", "base_terms": ["tollywood", "telugu film", "tamil film"]},
}

MOOD_QUERY_TERMS = {
    "happy": ["happy", "uplifting", "celebration"],
    "excited": ["energetic", "dance", "party"],
    "calm": ["calm", "chill", "soft"],
    "neutral": ["popular", "top songs", "fresh"],
    "sad": ["melancholy", "emotional", "slow"],
    "angry": ["intense", "power", "rock"],
    "anxious": ["relaxing", "soothing", "focus"],
    "stressed": ["stress relief", "soothing", "calm"],
    "tired": ["soft", "relax", "sleep"],
    "fearful": ["comfort", "calm", "ambient"],
    "surprised": ["trending", "viral", "popular"],
    "disgusted": ["relaxing", "clean mood", "ambient"],
}

FAMOUS_ALBUM_LIBRARY = {
    "global": {
        "label": "Global classics",
        "country": "US",
        "albums": [
            {"album_title": "Thriller", "artist": "Michael Jackson", "moods": ["happy", "excited", "surprised"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"},
            {"album_title": "Abbey Road", "artist": "The Beatles", "moods": ["calm", "neutral", "tired"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"},
            {"album_title": "Rumours", "artist": "Fleetwood Mac", "moods": ["sad", "calm", "neutral"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"},
            {"album_title": "21", "artist": "Adele", "moods": ["sad", "anxious", "calm"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3"},
            {"album_title": "Back in Black", "artist": "AC/DC", "moods": ["angry", "excited"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3"},
            {"album_title": "Future Nostalgia", "artist": "Dua Lipa", "moods": ["happy", "excited", "surprised"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3"},
        ],
    },
    "bollywood": {
        "label": "Bollywood albums",
        "country": "IN",
        "albums": [
            {"album_title": "Aashiqui 2", "artist": "Mithoon", "moods": ["sad", "calm", "anxious"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3"},
            {"album_title": "Rockstar", "artist": "A. R. Rahman", "moods": ["sad", "anxious", "calm"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3"},
            {"album_title": "Kal Ho Naa Ho", "artist": "Shankar-Ehsaan-Loy", "moods": ["calm", "sad", "neutral"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3"},
            {"album_title": "Kabir Singh", "artist": "Various Artists", "moods": ["sad", "anxious", "tired"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3"},
            {"album_title": "Om Shanti Om", "artist": "Vishal-Shekhar", "moods": ["happy", "excited", "surprised"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-11.mp3"},
            {"album_title": "Yeh Jawaani Hai Deewani", "artist": "Pritam", "moods": ["happy", "excited", "neutral"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-12.mp3"},
        ],
    },
    "tollywood": {
        "label": "Tollywood albums",
        "country": "IN",
        "albums": [
            {"album_title": "Baahubali: The Beginning", "artist": "M. M. Keeravani", "moods": ["excited", "surprised", "happy"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-13.mp3"},
            {"album_title": "RRR", "artist": "M. M. Keeravani", "moods": ["excited", "happy", "surprised"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-14.mp3"},
            {"album_title": "Pushpa: The Rise", "artist": "Devi Sri Prasad", "moods": ["excited", "angry", "happy"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"},
            {"album_title": "Ala Vaikunthapurramuloo", "artist": "Thaman S", "moods": ["happy", "excited", "neutral"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"},
            {"album_title": "Arjun Reddy", "artist": "Radhan", "moods": ["sad", "anxious", "calm"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"},
            {"album_title": "Rangasthalam", "artist": "Devi Sri Prasad", "moods": ["happy", "excited", "neutral"], "fallback_preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3"},
        ],
    },
}

FALLBACK_INDUSTRY_TRACKS = {
    "bollywood": [
        {
            "title": "Bollywood Mood Mix A",
            "artist": "FreeSource Studio",
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
            "external_url": "https://www.soundhelix.com/audio-examples",
            "artwork": "",
            "source": "soundhelix",
        },
        {
            "title": "Bollywood Mood Mix B",
            "artist": "FreeSource Studio",
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
            "external_url": "https://www.soundhelix.com/audio-examples",
            "artwork": "",
            "source": "soundhelix",
        },
    ],
    "hollywood": [
        {
            "title": "Hollywood Mood Mix A",
            "artist": "FreeSource Studio",
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-11.mp3",
            "external_url": "https://www.soundhelix.com/audio-examples",
            "artwork": "",
            "source": "soundhelix",
        },
        {
            "title": "Hollywood Mood Mix B",
            "artist": "FreeSource Studio",
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-12.mp3",
            "external_url": "https://www.soundhelix.com/audio-examples",
            "artwork": "",
            "source": "soundhelix",
        },
    ],
    "tollywood": [
        {
            "title": "Tollywood Mood Mix A",
            "artist": "FreeSource Studio",
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-13.mp3",
            "external_url": "https://www.soundhelix.com/audio-examples",
            "artwork": "",
            "source": "soundhelix",
        },
        {
            "title": "Tollywood Mood Mix B",
            "artist": "FreeSource Studio",
            "preview_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-14.mp3",
            "external_url": "https://www.soundhelix.com/audio-examples",
            "artwork": "",
            "source": "soundhelix",
        },
    ],
}


def _mood_terms(mood: str) -> list[str]:
    key = (mood or "neutral").lower()
    return MOOD_QUERY_TERMS.get(key, MOOD_QUERY_TERMS["neutral"])


def _normalize_itunes_results(results: list[dict]) -> list[dict[str, object]]:
    tracks: list[dict[str, object]] = []
    for row in results:
        preview = row.get("previewUrl") or ""
        external = row.get("trackViewUrl") or row.get("collectionViewUrl") or ""
        if not preview and not external:
            continue

        tracks.append(
            {
                "title": row.get("trackName") or row.get("collectionName") or "Untitled",
                "artist": row.get("artistName") or "Unknown Artist",
                "preview_url": preview,
                "external_url": external or preview,
                "artwork": row.get("artworkUrl100") or row.get("artworkUrl60") or "",
                "source": "itunes",
            }
        )
    return tracks


def _mood_score_album(album: dict[str, object], mood: str) -> tuple[int, int]:
    mood_key = (mood or "neutral").lower()
    moods = [str(item).lower() for item in album.get("moods", [])]
    return (1 if mood_key in moods else 0, -moods.index(mood_key) if mood_key in moods else 0)


def _fetch_album_card(section_key: str, album: dict[str, object], mood: str) -> dict[str, object]:
    config = FAMOUS_ALBUM_LIBRARY[section_key]
    query = f"{album['album_title']} {album['artist']}"
    cache_key = (section_key, str(album["album_title"]), str(mood or "neutral").lower())
    now_ts = time.time()

    cached = _INDUSTRY_RECO_CACHE.get(cache_key)
    if cached and float(cached.get("expires", 0)) > now_ts:
        return dict(cached.get("track", {}))

    preview_track: dict[str, object] = {}
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "music", "entity": "song", "country": config["country"], "limit": 5},
            timeout=8,
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                best = next((item for item in results if item.get("previewUrl")), results[0])
                preview_track = {
                    "album_title": album["album_title"],
                    "song_title": best.get("trackName") or album["album_title"],
                    "artist": best.get("artistName") or album["artist"],
                    "preview_url": best.get("previewUrl") or album["fallback_preview_url"],
                    "external_url": best.get("trackViewUrl") or best.get("collectionViewUrl") or "",
                    "artwork": best.get("artworkUrl100") or best.get("artworkUrl60") or "",
                    "source": "itunes",
                    "category": config["label"],
                    "mood": mood,
                }
    except Exception:
        preview_track = {}

    if not preview_track:
        preview_track = {
            "album_title": album["album_title"],
            "song_title": album["album_title"],
            "artist": album["artist"],
            "preview_url": album["fallback_preview_url"],
            "external_url": "",
            "artwork": "",
            "source": "curated",
            "category": config["label"],
            "mood": mood,
        }

    _INDUSTRY_RECO_CACHE[cache_key] = {"track": preview_track, "expires": now_ts + _INDUSTRY_RECO_TTL_SECONDS}
    return preview_track


def _album_recommendations_for_section(section_key: str, mood: str, limit: int = 4) -> list[dict[str, object]]:
    section = FAMOUS_ALBUM_LIBRARY[section_key]
    albums = sorted(section["albums"], key=lambda item: _mood_score_album(item, mood), reverse=True)
    return [_fetch_album_card(section_key, album, mood) for album in albums[:limit]]


def _album_recommendations(mood: str) -> dict[str, list[dict[str, object]]]:
    return {key: _album_recommendations_for_section(key, mood) for key in FAMOUS_ALBUM_LIBRARY}


def _fetch_itunes_industry_tracks(industry_key: str, mood: str, limit: int = 8) -> list[dict[str, object]]:
    config = INDUSTRY_QUERY_CONFIG[industry_key]
    mood_term = _mood_terms(mood)[0]
    base_term = config["base_terms"][0]
    query = f"{mood_term} {base_term}"
    cache_key = (industry_key, (mood or "neutral").lower())
    now_ts = time.time()

    cached = _INDUSTRY_RECO_CACHE.get(cache_key)
    if cached and float(cached.get("expires", 0)) > now_ts:
        return list(cached.get("tracks", []))

    tracks: list[dict[str, object]] = []
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "media": "music",
                "entity": "song",
                "country": config["country"],
                "limit": limit,
            },
            timeout=8,
        )
        if response.status_code == 200:
            data = response.json()
            tracks = _normalize_itunes_results(data.get("results", []))
    except Exception:
        tracks = []

    if not tracks:
        tracks = list(FALLBACK_INDUSTRY_TRACKS.get(industry_key, []))

    _INDUSTRY_RECO_CACHE[cache_key] = {
        "tracks": tracks,
        "expires": now_ts + _INDUSTRY_RECO_TTL_SECONDS,
    }
    return tracks


def _industry_recommendations(mood: str) -> dict[str, list[dict[str, object]]]:
    return _album_recommendations(mood)


def _ensure_track_catalog() -> None:
    if Track.objects.filter(is_active=True).exists():
        return
    for item in DEFAULT_TRACKS:
        Track.objects.update_or_create(
            track_id=item["track_id"],
            defaults={**item, "is_active": True},
        )


def _target_audio_profile(mood: str, intensity: float | None) -> tuple[float, float]:
    profile = MOOD_AUDIO_PROFILE.get((mood or "neutral").lower(), MOOD_AUDIO_PROFILE["neutral"])
    mood_intensity = max(0.0, min(float(intensity or 0.5), 1.0))

    target_valence = profile["valence"]
    target_energy = 0.55 * profile["energy"] + 0.45 * mood_intensity
    return target_valence, max(0.0, min(target_energy, 1.0))


def _score_track(track: Track, target_valence: float, target_energy: float) -> float:
    valence_distance = abs(track.valence - target_valence)
    energy_distance = abs(track.energy - target_energy)
    return (1.0 - (0.62 * valence_distance + 0.38 * energy_distance))


def _recommended_tracks_for_mood(mood: str, intensity: float | None, limit: int = 10) -> list[Track]:
    target_valence, target_energy = _target_audio_profile(mood, intensity)
    candidates = list(Track.objects.filter(is_active=True))
    candidates.sort(
        key=lambda track: _score_track(track, target_valence, target_energy),
        reverse=True,
    )
    return candidates[:limit]


def _serialize_track(track: Track) -> dict[str, object]:
    stream_url = track.stream_url
    preview_url = track.preview_url
    duration_seconds = track.duration_seconds

    if track.source == "local":
        stream_url = f"/music/stream/{track.id}/"
        preview_url = stream_url
        duration_seconds = min(duration_seconds or 45, 45)

    return {
        "id": track.id,
        "track_id": track.track_id,
        "title": track.title,
        "artist": track.artist,
        "source": track.source,
        "preview_url": preview_url,
        "stream_url": stream_url,
        "image_url": track.image_url,
        "valence": round(track.valence, 3),
        "energy": round(track.energy, 3),
        "duration_seconds": duration_seconds,
    }


def _ensure_music_session(request) -> MusicSession:
    if not request.session.session_key:
        request.session.save()
    session, _ = MusicSession.objects.get_or_create(
        user=request.user,
        session_key=request.session.session_key,
        defaults={"volume": 0.8, "playback_state": "paused", "queue_track_ids": []},
    )
    return session


def _build_state_payload(session: MusicSession, mood: str, intensity: float | None) -> dict[str, object]:
    recommended = _recommended_tracks_for_mood(mood, intensity, limit=10)
    queue_ids = [track.id for track in recommended]
    queue = [_serialize_track(track) for track in recommended]

    should_refresh_queue = session.current_track_id is None or not session.queue_track_ids
    if should_refresh_queue:
        session.queue_track_ids = queue_ids
        session.current_index = 0
        session.current_track = recommended[0] if recommended else None
        session.last_mood = mood or "neutral"
        session.save(update_fields=["queue_track_ids", "current_index", "current_track", "last_mood", "updated_at"])

    current_track = session.current_track
    if current_track is None and session.queue_track_ids:
        track_map = Track.objects.in_bulk(session.queue_track_ids)
        valid_queue = [track_map.get(track_id) for track_id in session.queue_track_ids if track_map.get(track_id)]
        if valid_queue:
            session.current_track = valid_queue[min(session.current_index, len(valid_queue) - 1)]
            session.save(update_fields=["current_track", "updated_at"])
            current_track = session.current_track

    return {
        "mood": mood,
        "intensity": round(float(intensity or 0.5), 3),
        "queue": queue,
        "album_recommendations": _album_recommendations(mood),
        "industry_recommendations": _industry_recommendations(mood),
        "current_index": session.current_index,
        "current_track": _serialize_track(current_track) if current_track else None,
        "playback_state": session.playback_state,
        "current_time_seconds": round(float(session.current_time_seconds or 0.0), 3),
        "volume": round(float(session.volume or 0.8), 3),
        "updated_at": session.updated_at.isoformat(),
    }


def _session_state_for_request(request) -> dict[str, object]:
    _ensure_track_catalog()
    recent_emotion = EmotionState.objects.filter(user=request.user).order_by("-detected_at").first()
    current_mood = getattr(recent_emotion, "emotion", "neutral")
    intensity = getattr(recent_emotion, "intensity", 0.5)
    session = _ensure_music_session(request)
    payload = _build_state_payload(session, current_mood, intensity)
    payload["recent_emotion"] = {
        "emotion": current_mood,
        "detected_at": getattr(recent_emotion, "detected_at", None).isoformat() if recent_emotion else None,
    }
    return payload


@login_required
def music_home(request):
    state = _session_state_for_request(request)
    return render(
        request,
        "music/index.html",
        {
            "current_mood": state["mood"],
            "recommended_tracks": state["queue"][:6],
            "album_recommendations": state.get("album_recommendations", {}),
            "player_state": json.dumps(state),
        },
    )


@login_required
@require_GET
def music_state(request):
    state = _session_state_for_request(request)
    return JsonResponse({"ok": True, "state": state})


def _track_wave_profile(track: Track) -> tuple[float, float, float]:
    base_frequency = 180 + int(track.valence * 220) + int(track.energy * 120)
    modulation = 0.5 + (track.energy * 0.7)
    amplitude = 0.2 + (track.valence * 0.25)
    return float(base_frequency), float(modulation), float(amplitude)


@login_required
@require_GET
def track_stream(request, track_id: int):
    try:
        track = Track.objects.get(pk=track_id, is_active=True)
    except Track.DoesNotExist as exc:
        raise Http404("Track not found") from exc

    sample_rate = 44100
    duration_seconds = 45 if track.source == "local" else max(15, min(int(track.duration_seconds or 45), 45))
    total_samples = sample_rate * duration_seconds
    base_frequency, modulation, amplitude = _track_wave_profile(track)
    phase_shift = (sum(ord(char) for char in track.track_id) % 360) / 57.2958

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for sample_index in range(total_samples):
            time_point = sample_index / sample_rate
            envelope = 0.55 + 0.45 * math.sin(2 * math.pi * modulation * 0.5 * time_point)
            tone_a = math.sin(2 * math.pi * base_frequency * time_point)
            tone_b = 0.45 * math.sin(2 * math.pi * (base_frequency * 0.5) * time_point + phase_shift)
            tone_c = 0.25 * math.sin(2 * math.pi * (base_frequency * 1.5) * time_point + phase_shift / 2)
            sample_value = int(32767 * amplitude * envelope * (tone_a + tone_b + tone_c) / 1.7)
            frames.extend(struct.pack("<h", max(-32768, min(32767, sample_value))))

        wav_file.writeframes(bytes(frames))

    response = HttpResponse(buffer.getvalue(), content_type="audio/wav")
    response["Content-Disposition"] = f'inline; filename="{track.track_id}.wav"'
    response["Cache-Control"] = "no-store"
    return response


def _normalize_volume(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.8
    return max(0.0, min(numeric, 1.0))


def _normalize_time_seconds(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, numeric)


@login_required
@require_POST
def player_action(request):
    session = _ensure_music_session(request)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}

    action = (payload.get("action") or "").lower().strip()
    queue_ids = session.queue_track_ids or []
    queue_tracks = Track.objects.in_bulk(queue_ids)
    queue = [queue_tracks.get(track_id) for track_id in queue_ids if queue_tracks.get(track_id)]

    if action == "play":
        session.playback_state = "playing"
    elif action == "pause":
        session.playback_state = "paused"
    elif action == "stop":
        session.playback_state = "stopped"
        session.current_time_seconds = 0.0
    elif action == "seek":
        session.current_time_seconds = _normalize_time_seconds(payload.get("current_time_seconds"))
    elif action == "volume":
        session.volume = _normalize_volume(payload.get("volume"))
    elif action in {"next", "previous"} and queue:
        if action == "next":
            session.current_index = (session.current_index + 1) % len(queue)
        else:
            session.current_index = (session.current_index - 1) % len(queue)
        session.current_track = queue[session.current_index]
        session.current_time_seconds = 0.0
    elif action == "select_track" and queue:
        track_id = payload.get("track_id")
        for index, track in enumerate(queue):
            if str(track.id) == str(track_id):
                session.current_index = index
                session.current_track = track
                session.current_time_seconds = 0.0
                break

    session.save()

    refreshed = _session_state_for_request(request)
    return JsonResponse({"ok": True, "state": refreshed})


def refresh_session_for_mood_shift(
    session: MusicSession,
    mood: str,
    intensity: float | None,
    cooldown_seconds: int = 45,
) -> dict[str, object]:
    now = timezone.now()
    mood_changed = (session.last_mood or "") != (mood or "")
    cooldown_over = True
    if session.last_mood_shift_at is not None:
        cooldown_over = (now - session.last_mood_shift_at).total_seconds() >= cooldown_seconds

    if mood_changed and cooldown_over:
        recommended = _recommended_tracks_for_mood(mood, intensity, limit=10)
        session.queue_track_ids = [track.id for track in recommended]
        session.current_index = 0
        session.current_track = recommended[0] if recommended else None
        session.current_time_seconds = 0.0
        session.last_mood = mood
        session.last_mood_shift_at = now
        session.save(
            update_fields=[
                "queue_track_ids",
                "current_index",
                "current_track",
                "current_time_seconds",
                "last_mood",
                "last_mood_shift_at",
                "updated_at",
            ]
        )

    return _build_state_payload(session, mood, intensity)
