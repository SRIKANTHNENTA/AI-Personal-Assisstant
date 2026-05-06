from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import render
from django.utils import timezone

from apps.chat_companion.models import Conversation, Message


@login_required
def notes_home(request):
    query = request.GET.get('q', '').strip()
    messages = Message.objects.filter(conversation__user=request.user).select_related('conversation').order_by('-timestamp')

    if query:
        messages = messages.filter(Q(content__icontains=query) | Q(conversation__title__icontains=query))

    note_cards = []
    for conversation in Conversation.objects.filter(user=request.user).order_by('-last_message_at')[:8]:
        latest_message = Message.objects.filter(conversation=conversation).order_by('-timestamp').first()
        preview = latest_message.content[:140] if latest_message else 'No notes captured yet.'
        timestamp = timezone.localtime(conversation.last_message_at).strftime('%b %d, %H:%M') if conversation.last_message_at else 'Now'
        note_cards.append({
            'conversation_id': conversation.id,
            'title': conversation.title or 'Untitled Thread',
            'preview': preview,
            'timestamp': timestamp,
            'tag': 'Memory link',
        })

    return render(request, 'notes/index.html', {
        'query': query,
        'messages': messages[:20],
        'note_cards': note_cards,
        'semantic_results': messages[:12],
    })
