"""
WebSocket Consumer for real-time chat
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from .models import Conversation, Message
from ai_services.cognitive_ai_integration import create_ai_services

logger = logging.getLogger(__name__)
User = get_user_model()


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        if not self.user.is_authenticated:
            await self.close()
            return
        
        self.room_name = f"chat_{self.user.id}"
        self.room_group_name = f"chat_{self.user.id}"
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Get or create active conversation
        self.conversation = await self.get_or_create_conversation()
    
    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        message_content = data.get('message', '')
        input_mode = data.get('input_mode', 'text')
        
        if not message_content:
            return
        
        # Save user message
        user_message = await self.save_message(
            message_type='user',
            content=message_content,
            input_mode=input_mode
        )
        
        # Analyze emotion from text
        emotion_data = await self.analyze_emotion(message_content)
        
        # Update message with emotion
        await self.update_message_emotion(user_message, emotion_data)
        
        # Generate AI response
        ai_response_data = await self.generate_ai_response(message_content)
        ai_response = ai_response_data.get('response', '')
        ai_explanation = ai_response_data.get('explanation', '')
        ai_source = ai_response_data.get('source', 'unknown')
        
        # Save AI message
        await self.save_message(
            message_type='assistant',
            content=ai_response,
            input_mode='text',
            explanation=ai_explanation,
        )
        
        # Send response to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'chat_message',
            'message': ai_response,
            'source': ai_source,
            'explanation': ai_explanation,
            'user_emotion': emotion_data.get('emotion', 'neutral'),
            'timestamp': str(user_message.timestamp)
        }))
    
    @database_sync_to_async
    def get_or_create_conversation(self):
        conversation, created = Conversation.objects.get_or_create(
            user=self.user,
            is_active=True,
            defaults={'title': 'New Conversation'}
        )
        return conversation
    
    @database_sync_to_async
    def save_message(self, message_type, content, input_mode, explanation=''):
        message = Message.objects.create(
            conversation=self.conversation,
            message_type=message_type,
            content=content,
            input_mode=input_mode,
            explanation=explanation or ''
        )
        
        # Update conversation message count
        self.conversation.message_count += 1
        self.conversation.save()
        
        return message
    
    @database_sync_to_async
    def analyze_emotion(self, text):
        services = create_ai_services(str(self.user.id))
        result = services.analyze_emotion(text=text)
        # Persist for dashboard visibility
        services.persist_emotion_state(result, source='text', text_content=text)
        return result
    
    @database_sync_to_async
    def update_message_emotion(self, message, emotion_data):
        if emotion_data.get('success'):
            emotion_info = emotion_data.get('fused_emotion', {})
            message.detected_emotion = emotion_info.get('emotion', 'neutral')
            message.sentiment_score = emotion_info.get('valence', 0.0)
            message.save()
    
    @database_sync_to_async
    def generate_ai_response(self, user_message):
        from ai_services.enhanced_response_handler import EnhancedResponseHandler
        
        # Use enhanced response handler for long-form responses with memory
        handler = EnhancedResponseHandler(str(self.user.id))
        
        # Get conversation history from database
        messages = Message.objects.filter(
            conversation=self.conversation
        ).order_by('-timestamp')[1:11]  # Last 10 turns for context
        
        history = [
            {
                'role': 'assistant' if msg.message_type == 'assistant' else 'user',
                'content': msg.content
            }
            for msg in reversed(messages)
        ]
        
        # Generate enhanced response with API keys and memory
        response_data = handler.generate_long_response(
            user_message=user_message,
            conversation_history=history,
            use_knowledge_base_fallback=True
        )
        
        response = response_data.get('response', 'I apologize, but I encountered an error generating a response. Please try again.')
        explanation = response_data.get('explanation', '')
        source = response_data.get('source', 'unknown')
        
        # Log response metadata for analytics
        logger.info(f"Response generated - Source: {source}, Length: {len(response)}")
        
        # Stateless mode: skip background conversational learning.

        return {
            'response': response,
            'explanation': explanation,
            'source': source,
        }
