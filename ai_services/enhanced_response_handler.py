"""
Enhanced AI Response Handler
- Uses API keys for longer, ChatGPT/Gemini-style detailed responses
- Maintains thread conversation memory/cache
- Handles sequential and random questions
- Falls back to knowledge base if offline
- Supports OpenAI, Gemini, and local knowledge base
"""

import os
import json
import logging
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from django.conf import settings
from ai_services.response_pipeline import LocalKnowledgeBaseMatcher, HybridResponsePipeline
from ai_services.gemini_handler import ask_gemini
from ai_services.llm_handler import ask_llm

logger = logging.getLogger(__name__)

# Thread-local conversation cache to maintain context across requests
_CONVERSATION_CACHE: Dict[str, Dict] = {}
_CACHE_LOCK = Lock()
CACHE_EXPIRY_MINUTES = 30
MAX_HISTORY_TURNS = 10
GENERIC_API_FALLBACK = "I am temporarily unable to generate an advanced response right now. Please try again in a moment."


class ConversationMemoryCache:
    """Manages conversation thread memory and context."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.cache_key = f"conv_{user_id}"
        self.history: List[Dict] = []
        self.last_updated = datetime.now()
        self.metadata = {
            'topic_chain': [],
            'sentiment_trend': [],
            'intent_sequence': []
        }
        self._load_from_cache()
    
    def _load_from_cache(self):
        """Load conversation from cache if not expired."""
        with _CACHE_LOCK:
            cached = _CONVERSATION_CACHE.get(self.cache_key)
            if cached:
                age = datetime.now() - cached.get('timestamp', datetime.now())
                if age < timedelta(minutes=CACHE_EXPIRY_MINUTES):
                    self.history = cached.get('history', [])[-MAX_HISTORY_TURNS:]
                    self.metadata = cached.get('metadata', self.metadata)
                    self.last_updated = cached.get('timestamp', datetime.now())
                    logger.info(f"Loaded conversation cache for user {self.user_id}")
                else:
                    _CONVERSATION_CACHE.pop(self.cache_key, None)
    
    def add_turn(self, user_message: str, assistant_response: str, intent: str = None, sentiment: float = 0.0):
        """Add a turn to conversation history."""
        self.history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat(),
            'intent': intent
        })
        self.history.append({
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        self.history = self.history[-MAX_HISTORY_TURNS * 2:]
        
        # Track metadata
        if intent:
            self.metadata['intent_sequence'].append(intent)
        self.metadata['sentiment_trend'].append(sentiment)
        
        self.last_updated = datetime.now()
        self._save_to_cache()
    
    def _save_to_cache(self):
        """Save conversation to cache."""
        with _CACHE_LOCK:
            _CONVERSATION_CACHE[self.cache_key] = {
                'history': self.history,
                'metadata': self.metadata,
                'timestamp': datetime.now()
            }
    
    def get_context_summary(self) -> str:
        """Generate a summary of conversation context for the AI."""
        if not self.history:
            return ""
        
        # Track topic progression
        intents = [h.get('intent') for h in self.history if h.get('intent')]
        topics = list(set(intents[-3:])) if intents else []
        
        summary = ""
        if len(self.history) > 2:
            summary += f"Recent conversation context:\n"
            # Show last 2-3 turns for context
            for msg in self.history[-4:]:
                role = "user" if msg['role'] == 'user' else "assistant"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                summary += f"- {role}: {content}\n"
        
        if topics:
            summary += f"\nTopics discussed: {', '.join(topics)}\n"
        
        return summary
    
    def clear(self):
        """Clear conversation cache."""
        with _CACHE_LOCK:
            _CONVERSATION_CACHE.pop(self.cache_key, None)
        self.history = []
        self.metadata = {'topic_chain': [], 'sentiment_trend': [], 'intent_sequence': []}


class EnhancedResponseHandler:
    """
    Enhanced AI response handler with API key support,
    conversation memory, and intelligent fallback.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.stateless_mode = str(getattr(settings, 'CHATBOT_STATELESS_MODE', True)).lower() in {'1', 'true', 'yes'}
        self.conversation_memory = ConversationMemoryCache(user_id)
        self.openai_api_key = settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else os.getenv('OPENAI_API_KEY', '')
        self.gemini_api_key = settings.GOOGLE_API_KEY if hasattr(settings, 'GOOGLE_API_KEY') else os.getenv('GOOGLE_API_KEY', '')
        self.kb_matcher = LocalKnowledgeBaseMatcher()
        self.hybrid_pipeline = HybridResponsePipeline(kb_matcher=self.kb_matcher)
        self.nlp_engine = self._init_nlp_engine()
        self.system_online = self._check_system_status()
    
    def _init_nlp_engine(self):
        """Initialize NLP engine for intent/sentiment detection."""
        try:
            from ai_services.nlp_engine import NLPEngine
            return NLPEngine()
        except Exception as e:
            logger.warning(f"NLP Engine init error: {e}")
            return None
    
    def _check_system_status(self) -> bool:
        """Check if API systems are online (has at least one API key)."""
        return bool(self.openai_api_key or self.gemini_api_key)
    
    def generate_long_response(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        system_prompt: str = None,
        use_knowledge_base_fallback: bool = True
    ) -> Dict:
        """
        Generate a long, detailed AI response with API keys.
        
        Returns:
        {
            'response': str,
            'source': 'openai' | 'gemini' | 'knowledge_base' | 'hybrid',
            'is_cached': bool,
            'metadata': {...}
        }
        """
        
        # Extract intent and sentiment
        intent = self._extract_intent(user_message)
        sentiment = self._analyze_sentiment(user_message)
        
        # Get conversation context (disabled in stateless mode)
        context = "" if self.stateless_mode else self.conversation_memory.get_context_summary()
        
        # Build system prompt for detailed responses
        detailed_system_prompt = system_prompt or self._build_system_prompt()
        
        response_data = {
            'response': '',
            'source': 'unknown',
            'is_cached': False,
            'explanation': '',
            'metadata': {
                'intent': intent,
                'sentiment': sentiment,
                'has_context': bool(context),
                'turn_count': len(self.conversation_memory.history)
            }
        }

        # Strategy 1: Normal LLM behavior via hybrid pipeline (KB first, API fallback).
        if use_knowledge_base_fallback:
            pipeline_result = self.hybrid_pipeline.get_response(
                user_input=user_message,
                conversation_history=conversation_history or ([] if self.stateless_mode else self.conversation_memory.history),
                user_context={'user_id': self.user_id},
                system_prompt=detailed_system_prompt,
            )

            base_response = pipeline_result.get('response', '')
            source = pipeline_result.get('source', 'hybrid')
            explanation = ''

            if self._is_knowledge_source(source):
                base_response, explanation = self._enhance_kb_answer_with_xai(
                    user_message=user_message,
                    answer=base_response,
                    source=source,
                    confidence=pipeline_result.get('confidence'),
                )

            response_data.update({
                'response': self._normalize_response_output(
                    user_message=user_message,
                    response=base_response,
                    intent=intent,
                ),
                'source': source,
                'explanation': explanation,
                'metadata': {
                    **response_data['metadata'],
                    'kb_confidence': pipeline_result.get('confidence'),
                    'intent': pipeline_result.get('intent', intent),
                },
            })

            if not self.stateless_mode:
                self.conversation_memory.add_turn(
                    user_message,
                    response_data['response'],
                    intent,
                    sentiment
                )
            return response_data

        # Strategy 2: API-only fallback when KB mode is disabled.
        if self.system_online:
            response_data['response'] = self._call_api_for_long_response(
                user_message,
                conversation_history or ([] if self.stateless_mode else self.conversation_memory.history),
                context,
                detailed_system_prompt
            )
            if response_data['response']:
                response_data['response'] = self._normalize_response_output(
                    user_message=user_message,
                    response=response_data['response'],
                    intent=intent,
                )
                response_data['source'] = 'api'
                return response_data
        
        # Strategy 3: Last resort graceful fallback
        response_data['response'] = self._normalize_response_output(
            user_message=user_message,
            response=self._graceful_fallback(user_message, intent),
            intent=intent,
        )
        response_data['source'] = 'fallback'
        return response_data

    def _is_knowledge_source(self, source: str) -> bool:
        source = (source or '').lower()
        return source in {
            'knowledge_base',
            'local_keyword_match',
            'local_semantic_match',
            'local_intent_classifier',
            'internet_knowledge',
        }

    def _enhance_kb_answer_with_xai(
        self,
        user_message: str,
        answer: str,
        source: str,
        confidence: Optional[float] = None,
    ) -> Tuple[str, str]:
        confidence_str = "unknown"
        if isinstance(confidence, (int, float)):
            confidence_str = f"{confidence:.2f}"

        explanation = (
            f"XAI: Retrieved from knowledge source '{source}'"
            f" with confidence {confidence_str}. "
            "The response was expanded to directly address the user question in plain language."
        )

        enhanced = answer.strip()
        if not enhanced:
            enhanced = "I found a related knowledge-base match but need more detail to answer accurately."

        # Always provide understandable detail instead of one-line KB output.
        word_count = len(enhanced.split())
        if word_count < 60:
            enhanced = (
                f"Direct answer:\n{enhanced}\n\n"
                "Detailed explanation:\n"
                f"You asked: {user_message.strip()}\n"
                "This response was selected from the local knowledge base and then expanded so the idea is easier to understand and apply. "
                "If you want, I can also provide examples, recent updates, or a comparison with related topics.\n\n"
                "How to go deeper:\n"
                "- Ask for a timeline or latest status\n"
                "- Ask for key facts in bullet points\n"
                "- Ask for practical examples"
            )

        return enhanced, explanation
    
    def _call_api_for_long_response(
        self,
        user_message: str,
        conversation_history: List[Dict],
        context: str,
        system_prompt: str
    ) -> str:
        """Call OpenAI or Gemini for long, detailed responses."""
        
        # Enhance history with context
        enhanced_history = conversation_history.copy() if conversation_history else []
        if context and len(enhanced_history) < 5:
            enhanced_history.append({
                'role': 'system',
                'content': f"Context from previous conversation:\n{context}"
            })
        
        # Try OpenAI first (if available)
        if self.openai_api_key:
            try:
                response = self._call_openai(
                    user_message,
                    enhanced_history,
                    system_prompt
                )
                if response:
                    return response
            except Exception as e:
                logger.warning(f"OpenAI call failed: {e}")
        
        # Fallback to Gemini (if available)
        if self.gemini_api_key:
            try:
                response = ask_gemini(
                    user_input=user_message,
                    history=enhanced_history,
                    system_prompt=system_prompt
                )
                if response and response != GENERIC_API_FALLBACK:
                    return response
            except Exception as e:
                logger.warning(f"Gemini call failed: {e}")
        
        return ""
    
    def _call_openai(
        self,
        user_message: str,
        conversation_history: List[Dict],
        system_prompt: str
    ) -> str:
        """Call OpenAI Chat Completions via HTTP to avoid SDK/httpx compatibility issues."""
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history (last 10 turns)
            for msg in conversation_history[-10:]:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            # Add current message
            messages.append({
                'role': 'user',
                'content': user_message
            })

            payload = {
                "model": os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                "messages": messages,
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                "top_p": 0.95,
            }

            req = urllib.request.Request(
                url="https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            choices = data.get("choices") or []
            if not choices:
                return ""

            content = choices[0].get("message", {}).get("content", "")
            return content.strip() if isinstance(content, str) else ""
        
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            logger.error(f"OpenAI API HTTP error: {e.code} - {err_body}")
            return ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""
    
    def _hybrid_knowledge_response(
        self,
        user_message: str,
        intent: str,
        conversation_history: List[Dict],
        context: str
    ) -> Dict:
        """Generate response using knowledge base + NLP hybrid approach."""

        # For open-domain questions, prefer the richer local explainer to avoid
        # accidental lexical KB matches with irrelevant canned responses.
        if intent in {'question', 'general', 'general_conversation'} or str(intent).startswith('general'):
            response = self._generate_nlp_response(user_message, intent, context)
            return {
                'response': response,
                'source': 'nlp',
                'is_cached': False,
                'metadata': {
                    'intent': intent,
                    'fallback': True
                }
            }
        
        # Try knowledge base match
        kb_result = self.kb_matcher.search(user_message)
        
        # Keep KB threshold conservative to avoid irrelevant lexical matches.
        if kb_result and kb_result.confidence > 0.82:
            response = kb_result.response
            source = "knowledge_base"
        else:
            # Generate using NLP pipeline
            response = self._generate_nlp_response(
                user_message,
                intent,
                context
            )
            source = "nlp"
        
        return {
            'response': response,
            'source': source,
            'is_cached': False,
            'metadata': {
                'intent': intent,
                'fallback': True
            }
        }
    
    def _generate_nlp_response(
        self,
        user_message: str,
        intent: str,
        context: str
    ) -> str:
        """Generate contextual response using local NLP."""

        message = (user_message or "").strip().lower()

        # Domain-focused local explanations so fallback remains helpful when APIs fail.
        if "physics" in message or "what is physics" in message or "explain physics" in message:
            return (
                "Physics is the branch of science that studies matter, energy, motion, forces, space, and time. "
                "It helps us understand how the universe works, from tiny particles to galaxies. "
                "In simple terms, physics explains why things move, fall, heat up, produce light, and interact with each other."
            )

        if any(k in message for k in ["oops", "oop", "object oriented"]):
            return (
                "Object-Oriented Programming (OOP) is a way of writing code using classes and objects. "
                "It makes programs easier to organize, reuse, and maintain using ideas like encapsulation, inheritance, "
                "polymorphism, and abstraction."
            )
        if "inheritance" in message:
            return (
                "Inheritance means a child class can reuse methods and properties from a parent class. "
                "This reduces repeated code and lets you extend behavior in a clean way."
            )

        if any(k in message for k in ["llm", "llms", "large language model", "chatgpt"]):
            return (
                "LLMs are Large Language Models trained on huge text datasets to understand and generate language. "
                "They are used for chat, writing, summarization, coding help, and question answering."
            )

        if message in {"python", "what is python", "explain python", "about python"}:
            return (
                "Python is a high-level, easy-to-read programming language used for web development, automation, "
                "data analysis, AI/ML, scripting, and backend services. It is popular because its syntax is simple, "
                "there are many libraries, and it helps you build things quickly."
            )

        if "python syllabus" in message or ("python" in message and "syllabus" in message):
            return (
                "A practical Python syllabus is: Basics (variables, loops, functions), Core (lists, dicts, files, exceptions), "
                "Intermediate (OOP, decorators, testing), and Specialization (Django, data science, or AI/ML). "
                "Build one mini-project after each phase."
            )

        if any(k in message for k in ["class and object", "class object", "what is class", "what is object"]):
            return (
                "A class is a blueprint that defines structure and behavior, and an object is a real instance of that class. "
                "For example, `Car` is a class and `my_car = Car('Toyota')` is an object."
            )

        if any(k in message for k in ["tablet", "tablets", "medicine", "medication", "medicines", "pill", "pills"]):
            return (
                "I can share general guidance, but I can’t prescribe medication. "
                "For fever, many adults use acetaminophen (paracetamol) or ibuprofen as over-the-counter options, "
                "following the label dosage and checking allergies, stomach/kidney issues, pregnancy, or other medicines first. "
                "If fever is high, lasts more than 24-48 hours, or you have warning signs like breathing trouble, chest pain, confusion, "
                "severe weakness, rash, or persistent vomiting, please contact a doctor urgently."
            )

        if any(k in message for k in ["fever", "temperature", "high temp", "chills"]):
            return (
                "I’m sorry you’re feeling unwell. For fever, rest, stay hydrated, and monitor your temperature every few hours. "
                "If your fever is high (around 102°F / 39°C or more), lasts over 24-48 hours, or you have warning signs "
                "like breathing trouble, chest pain, confusion, severe weakness, or persistent vomiting, please seek medical care promptly."
            )
        
        templates = {
            'greeting': [
                f"Hello! I'm here to help you. How can I assist you today?",
                f"Hi there! What can I help you with?",
            ],
            'task_create': [
                f"I can help you create a task. What would you like to add to your task list?",
                f"Sure, I'll help you create a new task. What are the details?",
            ],
            'task_query': [
                f"Let me help you check your tasks and prioritize them.",
                f"I can show you your current tasks and what's coming up.",
            ],
            'emotion_query': [
                f"Based on our conversation, I sense you're interested in understanding your emotional state. I'm here to help you reflect and process your feelings.",
                f"Let's talk about how you're feeling. I'm here to listen and support you.",
            ],
            'question': [
                "Could you share one more detail so I can answer precisely?",
                "Tell me what part you want first, and I will answer directly.",
            ]
        }
        
        intent_group = intent if intent in templates else 'question'
        base_responses = templates.get(intent_group, templates['question'])
        
        import random
        response = random.choice(base_responses)
        
        # Add context-aware extension without echoing the raw summary back to the user.
        if context:
            response += "\n\nI can connect this to what we've discussed earlier if you want a more personalized explanation."

        return response

    def _normalize_response_output(self, user_message: str, response: str, intent: str) -> str:
        """Strip accidental formatting artifacts so normal questions stay conversational."""
        if not response:
            return response

        text = response.strip()
        lower = text.lower()
        markers = [
            'statement:',
            'justification:',
            'rules:',
            'read the statement first',
            'scan the justification',
            'review the rules and example',
        ]

        if not any(marker in lower for marker in markers):
            return text

        cleaned = re.sub(r'(?im)^\s*(statement|justification|rules|example)\s*:\s*', '', text)
        cleaned = re.sub(
            r'(?im)^\s*-\s*(read the statement first|then scan the justification|then review the rules and example|keep the key idea first|add why it matters next|end with a practical example)\s*$',
            '',
            cleaned,
        )
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        if len(cleaned.split()) < 6:
            return self._graceful_fallback(user_message, intent)

        return cleaned

    def _format_structured_response(
        self,
        statement: str,
        justification: str,
        rules: List[str],
        example: Optional[str] = None,
    ) -> str:
        parts = [
            f"Statement:\n{statement}",
            f"\nJustification:\n{justification}",
            "\nRules:\n" + "\n".join(f"- {rule}" for rule in rules),
        ]
        if example:
            parts.append(f"\nExample:\n{example}")
        return "\n".join(parts)
    
    def _graceful_fallback(self, user_message: str, intent: str) -> str:
        """Provide graceful fallback response."""
        fallbacks = {
            'greeting': "Hello! I'm your AI assistant. How can I help?",
            'question': "That's an interesting question. I'm working to understand what you're asking. Can you provide more details?",
            'general': "I appreciate your message. I'm here to help and support you with whatever you need.",
        }
        
        import random
        return random.choice([fallbacks.get(intent, fallbacks['general'])])
    
    def _extract_intent(self, user_message: str) -> str:
        """Extract user intent from message."""
        if not self.nlp_engine:
            return "general"
        
        try:
            result = self.nlp_engine.extract_intent(user_message)
            return result.get('intent', 'general')
        except Exception:
            return "general"
    
    def _analyze_sentiment(self, user_message: str) -> float:
        """Analyze sentiment of user message."""
        if not self.nlp_engine:
            return 0.0
        
        try:
            sentiment = self.nlp_engine.analyze_sentiment(user_message)
            return sentiment.get('compound', 0.0)
        except Exception:
            return 0.0
    
    def _build_system_prompt(self) -> str:
        """Build detailed system prompt for long-form responses."""
        return """You are an empathetic, intelligent AI personal assistant designed to provide thoughtful, comprehensive support.

RESPONSE GUIDELINES:
1. Provide detailed, well-structured responses (3-5 paragraphs when appropriate)
2. Show understanding of context and ask clarifying questions when needed
3. Personalize responses based on conversation history
4. Be proactive in offering suggestions and support
5. Use clear formatting and bullet points where helpful
6. Balance warmth with professionalism
7. If a question is unclear, ask for clarification rather than guessing

PERSONALITY:
- Empathetic and supportive
- Clear and articulate
- Proactive and helpful
- Respectful of user autonomy
- Honest about limitations

Remember: You're not just answering questions - you're building a supportive relationship and understanding the user's needs context over time."""
