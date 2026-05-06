"""
Chat Companion Views
"""

import os
import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import Conversation, Message, UserMemory
from ai_services.nlp_engine import NLPEngine
from ai_services.speech_processor import SpeechProcessor
from ai_services.emotion_analyzer import EmotionAnalyzer
from ai_services.cognitive_ai_integration import create_ai_services


def _get_user_conversation_or_404(user, conversation_id):
    return Conversation.objects.get(id=conversation_id, user=user)


def _build_emotion_insight(conversation):
    latest_user_message = conversation.messages.filter(message_type='user').order_by('-timestamp').first()
    if not latest_user_message:
        return {
            'emotion': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'valence': 0.0,
            'arousal': 0.0,
            'explanation': 'No user message has been analyzed yet.',
            'message_preview': 'Waiting for your first message.',
        }

    return {
        'emotion': latest_user_message.detected_emotion or 'neutral',
        'sentiment_score': latest_user_message.sentiment_score or 0.0,
        'confidence': 0.0,
        'valence': latest_user_message.sentiment_score or 0.0,
        'arousal': 0.5,
        'explanation': latest_user_message.explanation or 'Emotion was detected from your latest message.',
        'message_preview': (latest_user_message.content or '')[:80],
    }


@login_required
def chat_view(request):
    """Main chat interface"""
    conversation_id = request.GET.get('conversation')

    if conversation_id:
        try:
            conversation = _get_user_conversation_or_404(request.user, conversation_id)
        except Conversation.DoesNotExist:
            conversation = None
    else:
        conversation = None

    if conversation is None:
        # Get or create active conversation
        conversation, created = Conversation.objects.get_or_create(
            user=request.user,
            is_active=True,
            defaults={'title': 'New Conversation'}
        )
    
    # Get recent messages (ordered by timestamp, oldest first for display)
    messages = conversation.messages.order_by('timestamp')[:50]
    
    # Get user memories
    memories = UserMemory.objects.filter(user=request.user).exclude(
        key__in=['name', 'user_name', 'username']
    ).order_by('-importance_score')[:10]

    emotion_insight = _build_emotion_insight(conversation)
    
    context = {
        'conversation': conversation,
        'messages': messages,
        'memories': memories,
        'emotion_insight': emotion_insight,
    }
    
    return render(request, 'chat_companion/chat.html', context)


@login_required
def conversation_thread(request, conversation_id):
    """Open a specific conversation thread."""
    conversation = _get_user_conversation_or_404(request.user, conversation_id)
    messages = conversation.messages.order_by('timestamp')[:50]
    memories = UserMemory.objects.filter(user=request.user).exclude(
        key__in=['name', 'user_name', 'username']
    ).order_by('-importance_score')[:10]

    emotion_insight = _build_emotion_insight(conversation)

    return render(request, 'chat_companion/chat.html', {
        'conversation': conversation,
        'messages': messages,
        'memories': memories,
        'emotion_insight': emotion_insight,
    })


@login_required
def new_conversation(request):
    """Create a new conversation"""
    # Mark current conversation as inactive
    Conversation.objects.filter(user=request.user, is_active=True).update(is_active=False)

    conversation = Conversation.objects.create(
        user=request.user,
        title='New Conversation',
        is_active=True,
    )

    return redirect('chat_companion:conversation_thread', conversation_id=conversation.id)


@login_required
@require_http_methods(["POST"])
def send_message(request):
    """Send a message and get AI response"""
    if request.method == 'POST':
        message_content = request.POST.get('message', '').strip()
        conversation_id = request.POST.get('conversation_id')
        
        if not message_content:
            return JsonResponse({'success': False, 'error': 'Empty message'})
        
        try:
            # Stateless mode safety: remove stale identity memories created earlier.
            UserMemory.objects.filter(
                user=request.user,
                key__in=['name', 'user_name', 'username']
            ).delete()

            # Get or create conversation
            if conversation_id:
                conversation = _get_user_conversation_or_404(request.user, conversation_id)
            else:
                conversation, created = Conversation.objects.get_or_create(
                    user=request.user,
                    is_active=True,
                    defaults={'title': message_content[:50] if message_content else 'New Conversation'}
                )
            
            # Save user message
            user_message = Message.objects.create(
                conversation=conversation,
                message_type='user',
                content=message_content,
                input_mode='text'
            )
            
            # Analyze emotion using Unified Cognitive AI
            ai_service = create_ai_services(str(request.user.id))
            emotion_result = ai_service.analyze_emotion(text=message_content)
            
            fused_emotion = emotion_result.get('fused_emotion', {})
            user_message.detected_emotion = fused_emotion.get('emotion', 'neutral')
            user_message.sentiment_score = fused_emotion.get('valence', 0.0)
            user_message.explanation = emotion_result.get('explanation', '')
            user_message.save()
            
            # Context-aware mode: use recent messages from the current conversation only.
            user_memories = {}
            history = [
                {
                    'role': 'assistant' if msg.message_type == 'assistant' else 'user',
                    'content': msg.content,
                    'timestamp': str(msg.timestamp),
                }
                for msg in conversation.messages.order_by('-timestamp')[:24]
            ]
            history.reverse()
            
            # Generate AI response using SMART SYSTEM with full history
            enriched_context = ai_service.get_enriched_chat_context()
            
            nlp_engine = NLPEngine()
            response_data = nlp_engine.generate_smart_response(
                user_message=message_content,
                conversation_history=history,
                user_context={
                    'user_id': str(request.user.id),
                    'current_emotion': enriched_context['emotion_state'].get('current_emotion')
                },
                user_memories=user_memories,
                system_prompt=None
            )
            
            ai_response_text = response_data.get('response', 'I am unable to respond at the moment.')
            response_source = response_data.get('source', 'unknown')
            used_api = response_data.get('used_api', False)
            learned_info = response_data.get('learned_info', {})
            
            # Stateless mode: do not persist learned personal memory.
            learned_info = {}
            
            # Save AI response with XAI insights
            ai_message = Message.objects.create(
                conversation=conversation,
                message_type='assistant',
                content=ai_response_text,
                input_mode='text',
                explanation=response_data.get('explanation', ''),
                cognitive_impact=response_data.get('cognitive_impact', {})
            )

            # Update conversation
            conversation.message_count += 2
            conversation.last_message_at = ai_message.timestamp
            conversation.save()
            
            return JsonResponse({
                'success': True,
                'user_message': {
                    'content': user_message.content,
                    'emotion': user_message.detected_emotion,
                    'sentiment_score': user_message.sentiment_score,
                    'explanation': user_message.explanation,
                    'timestamp': str(user_message.timestamp)
                },
                'emotion_insights': {
                    'emotion': user_message.detected_emotion,
                    'sentiment_score': user_message.sentiment_score,
                    'confidence': fused_emotion.get('confidence', 0.0),
                    'valence': fused_emotion.get('valence', 0.0),
                    'arousal': fused_emotion.get('arousal', 0.0),
                    'explanation': emotion_result.get('explanation', ''),
                    'message_preview': user_message.content[:80],
                },
                'ai_message': {
                    'content': ai_response_text,
                    'timestamp': str(ai_message.timestamp),
                    'source': response_source,
                    'explanation': ai_message.explanation,
                    'used_api': used_api,
                    'learned': list(learned_info.keys()) if learned_info else []
                }
            })
            
        except Exception as e:
            print(f"Error in send_message: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
@require_http_methods(["POST"])
def process_voice(request):
    """Process uploaded voice audio and return transcription"""
    if 'audio' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No audio file provided'})

    # Optional BCP-47 language code, e.g. en-US, hi-IN, es-ES.
    requested_language = (request.POST.get('language') or 'en-US').strip() or 'en-US'
    
    audio_file = request.FILES['audio']
    
    # Save temp file
    import tempfile
    from django.core.files.storage import default_storage
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
    
    try:
        # Detect format and convert if necessary
        from pydub import AudioSegment
        import imageio_ffmpeg as ffmpeg
        
        # Set ffmpeg path for pydub
        AudioSegment.converter = ffmpeg.get_ffmpeg_exe()
        
        # Load audio (pydub handles many formats automatically if ffmpeg is present)
        audio = AudioSegment.from_file(tmp_path)
        
        # Export as WAV (PCM 16-bit, 16kHz is ideal for Google STT)
        wav_path = tmp_path.replace('.wav', '_converted.wav')
        audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        
        processor = SpeechProcessor()
        result = processor.speech_to_text(wav_path, language=requested_language)
        
        # Clean up converted file
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': f"Processing Error: {str(e)}"})
    finally:
        # Final cleanup of the original temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@login_required
def conversation_history(request):
    """View conversation history"""
    conversations = Conversation.objects.filter(user=request.user).order_by('-last_message_at')
    
    context = {
        'conversations': conversations
    }
    
    return render(request, 'chat_companion/history.html', context)
