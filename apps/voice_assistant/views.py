from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone

import os
import tempfile

from ai_services.cognitive_ai_integration import CognitiveAIServices
from ai_services.speech_processor import SpeechProcessor
from ai_services.translator import MultilingualTranslator
from apps.emotion_tracker.models import EmotionState
from apps.task_manager.models import Task


@login_required
def voice_assistant_home(request):
    services = CognitiveAIServices(str(request.user.id))
    cognitive_state = services.get_cognitive_state()
    recent_emotion = EmotionState.objects.filter(user=request.user).order_by('-detected_at').first()
    translator = MultilingualTranslator()

    context = {
        'cognitive_state': cognitive_state,
        'recent_emotion': recent_emotion,
        'supported_languages': translator.supported_languages,
        'default_language': 'en',
        'voice_shortcuts': [
            'Summarize my day',
            'Read my latest note',
            'Check my schedule',
            'Open my inbox',
        ],
        'voice_feedback': [
            {'label': 'Transcription', 'value': 'Live speech-to-text ready'},
            {'label': 'Emotion', 'value': getattr(recent_emotion, 'emotion', 'steady').title()},
            {'label': 'Latency', 'value': 'Low'},
        ],
    }
    return render(request, 'voice_assistant/index.html', context)


def _voice_reply_for_text(user, text: str) -> str:
    message = (text or '').strip()
    lowered = message.lower()

    if not message:
        return 'I did not catch that. Please try again.'

    pending_tasks = Task.objects.filter(user=user, status='pending').count()
    today_tasks = Task.objects.filter(
        user=user,
        due_date__date=timezone.localdate(),
        status__in=['pending', 'in_progress']
    ).count()

    latest_emotion = EmotionState.objects.filter(user=user).order_by('-detected_at').first()
    emotion_name = getattr(latest_emotion, 'emotion', 'steady')

    if 'summarize my day' in lowered or 'summary' in lowered:
        return f"Today you have {today_tasks} active tasks and {pending_tasks} pending tasks in total. Your latest mood signal is {emotion_name}."

    if 'check my schedule' in lowered or 'schedule' in lowered or 'tasks' in lowered:
        return f"You currently have {pending_tasks} pending tasks. {today_tasks} of them are scheduled for today."

    if 'read my latest note' in lowered or 'latest note' in lowered:
        return 'Your notes module is ready. Open Notes to review your latest saved entry.'

    if 'open my inbox' in lowered or 'inbox' in lowered or 'email' in lowered:
        return 'Your inbox module is available. Open Email AI to check and draft messages.'

    if 'mood' in lowered or 'emotion' in lowered or 'how am i feeling' in lowered:
        return f"Your latest detected mood is {emotion_name}."

    if 'help' in lowered or 'what can you do' in lowered:
        return 'I can summarize your day, check schedule status, report your latest mood, and help with voice commands for your assistant modules.'

    return (
        'Voice Assistant received your command. I can help with schedule checks, mood readout, '
        'and quick module actions. Try: summarize my day.'
    )


@login_required
@require_http_methods(["POST"])
def language_bridge(request):
    """
    Detects language and translates text.
    Used by /voice page for multilingual voice input/output.
    """
    text = (request.POST.get('text') or '').strip()
    target_language = (request.POST.get('target_language') or 'en').strip().lower()
    source_language = (request.POST.get('source_language') or 'auto').strip().lower()

    if not text:
        return JsonResponse({'success': False, 'error': 'Empty text'})

    translator = MultilingualTranslator()

    detection = translator.detect_language(text)
    detected_code = detection.get('language_code', 'en') if detection.get('success') else 'en'
    detected_name = detection.get('language_name', 'English') if detection.get('success') else 'English'

    # Keep source explicit when manual mode is requested.
    effective_source = detected_code if source_language == 'auto' else source_language

    translation = translator.translate_text(
        text=text,
        target_lang=target_language,
        source_lang=effective_source,
    )

    if not translation.get('success'):
        return JsonResponse({
            'success': True,
            'translation_fallback': True,
            'error': translation.get('error', 'Translation failed'),
            'detected_language': detected_code,
            'detected_language_name': detected_name,
            'source_language': effective_source,
            'target_language': target_language,
            'translated_text': text,
            'original_text': text,
        })

    return JsonResponse({
        'success': True,
        'detected_language': detected_code,
        'detected_language_name': detected_name,
        'source_language': translation.get('source_language', effective_source),
        'target_language': target_language,
        'translated_text': translation.get('translated_text', text),
        'original_text': text,
    })


@login_required
@require_http_methods(["POST"])
def process_command(request):
    """Handle voice assistant commands without routing through chat module."""
    text = (request.POST.get('text') or '').strip()
    if not text:
        return JsonResponse({'success': False, 'error': 'Empty command'})

    reply = _voice_reply_for_text(request.user, text)
    return JsonResponse({'success': True, 'reply': reply})


@login_required
@require_http_methods(["POST"])
def transcribe_voice(request):
    """Transcribe uploaded voice audio for the voice assistant only."""
    if 'audio' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No audio file provided'})

    requested_language = (request.POST.get('language') or 'en-US').strip() or 'en-US'
    audio_file = request.FILES['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        from pydub import AudioSegment
        import imageio_ffmpeg as ffmpeg

        AudioSegment.converter = ffmpeg.get_ffmpeg_exe()
        audio = AudioSegment.from_file(tmp_path)
        wav_path = tmp_path.replace('.wav', '_converted.wav')
        audio.export(wav_path, format='wav', parameters=['-ac', '1', '-ar', '16000'])

        processor = SpeechProcessor()
        result = processor.speech_to_text(wav_path, language=requested_language)

        if os.path.exists(wav_path):
            os.remove(wav_path)

        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({'success': False, 'error': f'Processing Error: {exc}'})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
