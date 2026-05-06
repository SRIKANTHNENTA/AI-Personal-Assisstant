"""
NLP Engine for natural language processing and understanding
Uses OpenAI GPT-4 and transformers for intent recognition and sentiment analysis
"""

import os
from typing import Dict, List, Optional
from django.conf import settings
import openai
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from ai_services.response_pipeline import HybridResponsePipeline

# Set seed for consistent language detection
DetectorFactory.seed = 0


class NLPEngine:
    """Main NLP processing engine"""
    
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_MODEL
        openai.api_key = self.openai_api_key
        self.vader = SentimentIntensityAnalyzer()
        self.hybrid_pipeline = HybridResponsePipeline()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER
        Returns: dict with compound, positive, negative, neutral scores
        """
        try:
            scores = self.vader.polarity_scores(text)
            return {
                'compound': scores['compound'],  # -1 to 1
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def detect_emotion_from_text(self, text: str) -> Dict[str, any]:
        """
        Detect emotion from text using sentiment analysis
        Returns: emotion label and confidence
        """
        sentiment = self.analyze_sentiment(text)
        compound = sentiment['compound']
        
        # Map sentiment to emotions
        if compound >= 0.5:
            emotion = 'happy'
            intensity = min(compound, 1.0)
        elif compound >= 0.1:
            emotion = 'calm'
            intensity = compound
        elif compound >= -0.1:
            emotion = 'neutral'
            intensity = 0.5
        elif compound >= -0.5:
            emotion = 'sad'
            intensity = abs(compound)
        else:
            emotion = 'angry'
            intensity = min(abs(compound), 1.0)
        
        return {
            'emotion': emotion,
            'intensity': intensity,
            'confidence': abs(compound),
            'sentiment_scores': sentiment
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            return detect(text)
        except:
            return 'en'
    
    def extract_intent(self, text: str) -> Dict[str, any]:
        """
        Extract user intent from text
        Returns: intent category and confidence
        """
        text_lower = text.lower()
        
        # Simple keyword-based intent detection
        intents = {
            'task_create': ['create task', 'add task', 'new task', 'remind me', 'schedule'],
            'task_query': ['show tasks', 'list tasks', 'what tasks', 'my tasks'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'take care'],
            'question': ['what', 'when', 'where', 'who', 'why', 'how'],
            'emotion_query': ['how am i', 'my mood', 'feeling'],
            'help': ['help', 'assist', 'support'],
        }
        
        detected_intent = 'general_conversation'
        confidence = 0.5
        
        for intent, keywords in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_intent = intent
                confidence = 0.8
                break
        
        return {
            'intent': detected_intent,
            'confidence': confidence
        }
    
    def generate_response(self, 
                         user_message: str, 
                         conversation_history: List[Dict] = None,
                         user_context: Dict = None,
                         system_prompt: str = None) -> str:
        """
        Generate AI response using OpenAI GPT
        """
        if not self.openai_api_key or self.openai_api_key == '':
            return "I'm sorry, I need an OpenAI API key to respond. Please configure it in your settings."
        
        try:
            # Initialize OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Build conversation context
            base_system_prompt = system_prompt or """You are a helpful, empathetic AI personal assistant. 
                    You help users manage their tasks, understand their emotions, and provide 
                    personalized support. Be friendly, understanding, and proactive in offering help.
                    Keep responses concise but warm."""
            
            messages = [
                {
                    "role": "system",
                    "content": base_system_prompt
                }
            ]
            
            # Add user context if available
            if user_context:
                context_str = "User context: stateless session"
                if user_context.get('current_emotion'):
                    context_str += f", Current mood: {user_context['current_emotion']}"
                messages[0]['content'] += f"\n{context_str}"
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages
                    messages.append({
                        "role": msg.get('role', 'user'),
                        "content": msg.get('content', '')
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Call OpenAI API with new client
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            print(f"OpenAI API error: {e}")
            
            # Check if it's a quota error
            if 'insufficient_quota' in error_str or '429' in error_str:
                return self._get_fallback_response(user_message)
            elif 'model_not_found' in error_str or '404' in error_str:
                return "The AI model is not available. Please check your OpenAI account settings."
            else:
                return f"I'm having trouble connecting right now. Error: {error_str}"
    
    def _get_fallback_response(self, user_message: str) -> str:
        """
        Provide intelligent fallback responses when OpenAI is unavailable
        """
        message_lower = user_message.lower()
        
        # Greeting responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            return "Hello! I'm your AI assistant. I'd love to help you, but my OpenAI quota has been exceeded. Please add billing to your OpenAI account at https://platform.openai.com/account/billing to enable AI responses. In the meantime, I can still track your tasks and emotions!"
        
        # Task-related
        if any(word in message_lower for word in ['task', 'todo', 'remind']):
            return "I can help you manage tasks! However, my AI brain needs OpenAI credits to provide smart responses. Please add billing to your OpenAI account. You can still create tasks manually through the Tasks page."
        
        # Emotion-related
        if any(word in message_lower for word in ['feel', 'mood', 'emotion', 'sad', 'happy']):
            return "I'm here to support you emotionally! Unfortunately, my AI capabilities are limited right now due to OpenAI quota limits. Please add credits to your OpenAI account to unlock full emotional intelligence features."
        
        # Help
        if 'help' in message_lower:
            return "I'm your personal AI assistant! I can help with tasks, emotional support, and daily planning. However, I need OpenAI API credits to function. Please visit https://platform.openai.com/account/billing to add billing to your account."
        
        # Default fallback
        return "Thanks for your message! I'd love to respond intelligently, but my OpenAI quota has been exceeded. Please add billing to your OpenAI account at https://platform.openai.com/account/billing. The application is working perfectly - it just needs API credits to enable AI responses!"
    
    def _get_template_response(self, user_message: str) -> Optional[str]:
        """
        Check if user message matches pre-loaded templates for basic responses
        Returns template response if matched, None otherwise
        """
        import json
        import random
        from pathlib import Path
        
        try:
            # Load response templates
            template_path = Path(__file__).parent / 'response_templates.json'
            with open(template_path, 'r') as f:
                templates = json.load(f)
            
            message_lower = user_message.lower().strip()
            
            # Special handling for task/meeting requests
            task_keywords = ['schedule', 'meeting', 'arrange', 'appointment', 'remind', 'task']
            if any(keyword in message_lower for keyword in task_keywords):
                task_templates = templates.get('task_creation', {})
                if task_templates.get('responses'):
                    return random.choice(task_templates['responses'])
            
            # Check each template category
            for category, data in templates.items():
                patterns = data.get('patterns', [])
                responses = data.get('responses', [])
                
                # Check if any pattern matches
                for pattern in patterns:
                    if pattern in message_lower:
                        # Return random response from this category
                        return random.choice(responses)
            
            return None
            
        except Exception as e:
            print(f"Template loading error: {e}")
            return None
    
    def get_response(self,
                     user_input: str,
                     conversation_history: List[Dict] = None,
                     user_context: Dict = None,
                     system_prompt: str = None) -> Dict[str, any]:
        """
        Hybrid production response pipeline.
        Order: local knowledge base/classifier -> external LLM -> graceful fallback.
        """
        return self.hybrid_pipeline.get_response(
            user_input=user_input,
            conversation_history=conversation_history or [],
            user_context=user_context or {},
            system_prompt=system_prompt,
        )

    def generate_smart_response(self,
                               user_message: str,
                               conversation_history: List[Dict] = None,
                               user_context: Dict = None,
                               user_memories: Dict = None,
                               system_prompt: str = None) -> Dict[str, any]:
        """
        Smart response orchestration for chat:
        1) learning extraction from user profile statements
        2) local deterministic router
        3) hybrid response pipeline (local intents/classifier -> LLM -> graceful fallback)
        """
        from ai_services.smart_router import smart_router

        # Stateless mode: do not learn/store user identity or use memory-bound personalization.
        user_memories = {}

        local_response = smart_router(user_message, user_memories)
        if local_response:
            return {
                'response': local_response,
                'source': 'local_router',
                'used_api': False
            }

        # Use the enhanced long-form handler for the main chat path so the UI
        # gets ChatGPT-style detailed answers rather than short KB snippets.
        try:
            from ai_services.enhanced_response_handler import EnhancedResponseHandler

            handler_user_id = str((user_context or {}).get('user_id') or 'system')
            handler = EnhancedResponseHandler(handler_user_id)

            response_data = handler.generate_long_response(
                user_message=user_message,
                conversation_history=conversation_history or [],
                system_prompt=system_prompt,
                use_knowledge_base_fallback=True,
            )

            return {
                'response': response_data.get('response', ''),
                'source': response_data.get('source', 'enhanced_handler'),
                'used_api': response_data.get('source') in {'api', 'openai', 'gemini'},
                'explanation': response_data.get('explanation', ''),
                'learned_info': {},
                'metadata': response_data.get('metadata', {}),
            }
        except Exception as exc:
            print(f"Enhanced response handler error: {exc}")

        full_context = self._build_full_context(
            user_message,
            conversation_history or [],
            user_context or {},
            user_memories,
        )
        agent_system_prompt = self._get_agent_system_prompt(full_context)

        pipeline_response = self.get_response(
            user_input=user_message,
            conversation_history=conversation_history or [],
            user_context=user_context or {},
            system_prompt=agent_system_prompt or system_prompt,
        )

        pipeline_response.setdefault('learned_info', {})
        return pipeline_response

    def _build_full_context(self, message: str, history: List[Dict], context: Dict, memories: Dict) -> Dict:
        """Gathers all available sensory and memory data into a structured context"""
        from ai_services.cognitive_ai_integration import create_ai_services
        
        # Get real-time emotion state
        emotion_state = {}
        try:
            # Use the current user context when available so we reuse cached services.
            user_id = context.get('user_id', 'system')
            cognitive = create_ai_services(str(user_id))
            state = cognitive.get_cognitive_state()
            emotion_state = {
                'current_emotion': state.get('current_emotion', 'neutral'),
                'arousal': state.get('arousal', 0.5),
                'valence': state.get('valence', 0.0),
                'stress_level': state.get('stress_level', 0.2)
            }
        except:
            emotion_state = {'current_emotion': 'neutral', 'arousal': 0.5, 'valence': 0.0}

        return {
            'message': message,
            'history_count': len(history or []),
            'user_info': {
                'name': "User",
                'memories': list(memories.keys())
            },
            'emotion_state': emotion_state
        }
    def _get_agent_system_prompt(self, context: Dict) -> str:
        """Constructs a master system prompt for the Agentic Brain based on advanced behavioral rules."""
        emotion = context['emotion_state']['current_emotion']
        valence = context['emotion_state']['valence']
        arousal = context['emotion_state']['arousal']
        
        # Determine adapter mood based on valence/arousal
        if valence < -0.3:
            mood_instruction = "The user appears stressed or sad. Respond with deep empathy, reassurance, and a supportive tone."
        elif valence > 0.3:
            mood_instruction = "The user appears happy. Respond in a very friendly, enthusiastic, and engaging tone."
        else:
            mood_instruction = "The user is in a neutral state. Respond in a professional and helpful tone."

        return f"""You are a highly intelligent, emotionally-aware personal AI assistant.

YOUR CAPABILITIES:
* Natural conversation like ChatGPT
* Context awareness and memory usage
* Emotion-aware responses based on user state
* Helping with coding, academics, problem-solving, and general queries

CURRENT USER CONTEXT:
* Detected Emotion: {emotion}
* Emotion Valence: {valence} (Arousal: {arousal})
* **MOOD ADAPTATION**: {mood_instruction}

YOUR BEHAVIOR RULES:
1. Emotion Awareness: Respond exactly according to the "MOOD ADAPTATION" instruction above.
2. Intelligence: Provide accurate, detailed, and helpful answers. For coding, give clean, working code. For explanations, give clear, structured answers.
3. Statelessness: Treat every message independently and only answer the asked question.
4. Efficiency: Avoid unnecessary long explanations unless asked. Be clear, direct, and useful.
5. Safety: Do not generate harmful or unsafe content. If unsure, say you don’t know instead of guessing.

RESPONSE STYLE:
* Natural human-like conversation.
* Slight emotional adaptation based on detected mood.
* No robotic or repetitive phrases.
* For normal questions, respond directly in plain natural language.
* Use bullets only when they genuinely improve clarity.
* Output only the final answer. Do not mention internal analysis.
"""

    def _execute_tool(self, name: str, input_str: str, memories: Dict) -> Optional[str]:
        """Executes a specific internal capability"""
        if name == 'GET_USER_NAME':
            return "I don't persist personal identity between messages."
        
        if name == 'SEARCH_KNOWLEDGE':
            from ai_services.conversation_learner import ConversationLearner
            learner = ConversationLearner()
            return learner.find_learned_response(input_str)
            
        return None

    def _safety_fallback(self, message: str, memories: Dict) -> Dict[str, any]:
        """Legacy rule-based fallback if the LLM fails"""
        if 'name' in message.lower():
            return {'response': "I don't store personal names. Ask me your current question and I'll answer directly.", 'source': 'safety_rule', 'used_api': False}
        return {'response': "I am here to help, but I'm having trouble connecting to my brain right now.", 'source': 'safety_error', 'used_api': False}
    
    def _extract_and_learn(self, user_message: str, user_memories: Dict) -> Optional[Dict]:
        """
        Extract personal information from user message and generate learning response
        Returns dict with 'response' and 'extracted' info, or None if nothing learned
        """
        # Stateless behavior: disable personal-info extraction and memory learning.
        return None

        import json
        import re
        import random
        from pathlib import Path
        
        try:
            template_path = Path(__file__).parent / 'learning_patterns.json'
            with open(template_path, 'r') as f:
                patterns = json.load(f)
            
            message_lower = user_message.lower().strip()
            extracted_info = {}
            
            # Check personal info patterns
            personal_patterns = patterns.get('personal_info_patterns', {})
            
            # Name extraction
            if 'name' in personal_patterns:
                for pattern in personal_patterns['name']['patterns']:
                    # Convert pattern to regex
                    regex_pattern = pattern.replace('{name}', r'(\w+)')
                    match = re.search(regex_pattern, message_lower)
                    if match:
                        name = match.group(1).capitalize()
                        extracted_info['name'] = name
                        response_template = random.choice(personal_patterns['name']['response_templates'])
                        return {
                            'response': response_template.replace('{name}', name),
                            'extracted': extracted_info
                        }
            
            # Activity/hobby extraction
            activity_patterns = patterns.get('activity_tracking', {})
            for pattern in activity_patterns.get('patterns', []):
                regex_pattern = pattern.replace('{activity}', r'(.+)')
                match = re.search(regex_pattern, message_lower)
                if match:
                    activity = match.group(1).strip()
                    extracted_info['activity'] = activity
                    response_template = random.choice(activity_patterns['response_templates'])
                    return {
                        'response': response_template.replace('{activity}', activity),
                        'extracted': extracted_info
                    }
            
            # Location extraction
            if 'location' in personal_patterns:
                for pattern in personal_patterns['location']['patterns']:
                    regex_pattern = pattern.replace('{location}', r'(.+)')
                    match = re.search(regex_pattern, message_lower)
                    if match:
                        location = match.group(1).strip().title()
                        extracted_info['location'] = location
                        response_template = random.choice(personal_patterns['location']['response_templates'])
                        return {
                            'response': response_template.replace('{location}', location),
                            'extracted': extracted_info
                        }
            
            return None
            
        except Exception as e:
            print(f"Learning extraction error: {e}")
            return None
    
    def _get_contextual_template_response(self, user_message: str, user_memories: Dict) -> Optional[str]:
        """
        Get template response with personalization based on user memories
        """
        import random
        
        # First try basic template matching
        basic_response = self._get_template_response(user_message)
        
        if basic_response:
            return basic_response
        
        return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        Simple implementation - can be enhanced with spaCy
        """
        # This is a placeholder - implement with spaCy for better results
        entities = {
            'dates': [],
            'times': [],
            'persons': [],
            'locations': [],
            'tasks': []
        }
        
        # Simple pattern matching for dates/times
        import re
        
        # Extract dates (simple patterns)
        date_patterns = r'\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        dates = re.findall(date_patterns, text.lower())
        entities['dates'] = list(set(dates))
        
        # Extract times
        time_patterns = r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b'
        times = re.findall(time_patterns, text.lower())
        entities['times'] = list(set(times))
        
        return entities
