"""
New conversation view - creates a new conversation and redirects to chat
"""
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from apps.chat_companion.models import Conversation

@login_required
def new_conversation(request):
    """Create a new conversation"""
    # Mark current conversation as inactive
    Conversation.objects.filter(user=request.user, is_active=True).update(is_active=False)
    
    # Redirect to chat - it will create a new active conversation
    return redirect('chat_companion:chat')
