# Enhanced AI Response System

## Overview
The Enhanced AI Response Handler provides intelligent, context-aware responses with API key support, conversation memory caching, and intelligent fallbacks.

## Features

### 1. **API Key Integration**
- **OpenAI GPT-4/GPT-3.5** — For long-form, detailed responses
- **Google Gemini** — Advanced reasoning and alternatives
- **Automatic Fallback** — If APIs are unavailable, uses knowledge base

### 2. **Long-Form Responses**
- Generates 2000-token detailed responses (like ChatGPT/Gemini)
- Structured, multi-paragraph answers
- Context-aware and personalized
- Higher max_tokens for comprehensive explanations

### 3. **Conversation Memory Cache**
- **Thread-local caching** — Maintains conversation context across requests
- **30-minute expiry** — Auto-clears old conversations
- **Metadata tracking** — Monitors intent sequences and sentiment trends
- **Smart context injection** — Previous topics inform new responses

### 4. **Intelligent Question Handling**
- **Sequential questions** — Understands topic progression
- **Random questions** — Handles unexpected topics gracefully
- **Intent detection** — Classifies user intent (task, emotion, greeting, etc.)
- **Sentiment analysis** — Adapts tone based on user emotional state

### 5. **Offline Fallback Strategy**
- **Knowledge base** — Uses local intents JSON when APIs offline
- **NLP generation** — Creates contextual responses from templates
- **Graceful degradation** — Never breaks, always replies

## Configuration

### Step 1: Set Up API Keys

Add to your `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4-turbo

# Google Gemini Configuration
GOOGLE_API_KEY=AIza-your-api-key-here
GEMINI_MODEL=gemini-1.5-pro
```

### Step 2: Get API Keys

#### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to `.env` as `OPENAI_API_KEY`

#### Google Gemini
1. Go to https://ai.google.dev/
2. Get API key for free
3. Add to `.env` as `GOOGLE_API_KEY`

### Step 3: Restart Server
```bash
python manage.py runserver
```

## How It Works

### Response Generation Pipeline

```
User Message
    ↓
[Extract Intent & Sentiment]
    ↓
[Check API Availability]
    ↓
Yes: Call OpenAI/Gemini → Long Response (2000 tokens)
    ↓
No: Use Knowledge Base + NLP → Contextual Template Response
    ↓
[Add to Conversation Memory Cache]
    ↓
Response Sent to User
```

### Conversation Memory Example

```python
# First turn
User: "I'm feeling anxious about my project deadline"
AI: [Detailed response with empathy - 300+ words]
     [Cached: intent=concern, sentiment=-0.4]

# Second turn (related)
User: "How can I break it into smaller tasks?"
AI: [References previous anxiety, provides structured breakdown]
     [Uses cached intent sequence to understand context]

# Third turn (random topic)
User: "What's the weather today?"
AI: [Handles gracefully with knowledge base fallback]
     [Still remembers project context if brought up again]
```

## Response Sources

### 1. **API Source** (Preferred - 2000 tokens)
- OpenAI GPT-4: Detailed reasoning
- Gemini: Advanced analysis
- Requires API key + internet

### 2. **Hybrid Source** (Fallback)
- Knowledge base intents (local)
- Template-based responses (rule-based)
- NLP context injection
- Works offline or when APIs offline

### 3. **Fallback Source** (Last Resort)
- Graceful generic responses
- Never leaves user without answer
- Indicates limitations if needed

## Intent Detection

The system automatically detects user intent:
- **greeting** — "Hi", "Hello", "Hey"
- **task_create** — "Create task", "Schedule", "Remind me"
- **task_query** — "Show tasks", "What's due?"
- **emotion_query** — "How am I feeling?", "My mood"
- **question** — "What", "When", "Where", "How"
- **general** — Default for unclassified

## Sentiment Analysis

Sentiment scores range from -1 (negative) to +1 (positive):
- **≥ 0.5** → Happy/Enthusiastic responses
- **0.1 to 0.5** → Calm/Supportive responses
- **-0.1 to 0.1** → Neutral tone
- **-0.5 to -0.1** → Sad/Empathetic responses
- **< -0.5** → Concerned/Supportive with action

## Usage Examples

### In Chat Consumer
```python
from ai_services.enhanced_response_handler import EnhancedResponseHandler

handler = EnhancedResponseHandler(user_id)
response_data = handler.generate_long_response(
    user_message="Tell me about project management",
    conversation_history=history,  # Previous messages
    use_knowledge_base_fallback=True
)

# response_data contains:
# - 'response': str (the actual response)
# - 'source': 'openai' | 'gemini' | 'hybrid' | 'fallback'
# - 'metadata': intent, sentiment, context info
```

### Direct Import
```python
from ai_services.enhanced_response_handler import ConversationMemoryCache

# Maintain per-user conversation memory
cache = ConversationMemoryCache(user_id="123")
cache.add_turn("What's a task?", "A task is...", intent="question")
cache.get_context_summary()  # Get conversation context
```

## Environment Variables

```env
# API Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo
GOOGLE_API_KEY=AIza-...
GEMINI_MODEL=gemini-1.5-pro

# Cache Settings
CONVERSATION_CACHE_EXPIRY=30  # minutes

# NLP Settings
ENABLE_NEURAL_INTENT_CLASSIFIER=true
```

## Troubleshooting

### Issue: Responses are short
**Solution:** Check API keys are configured. If offline, responses use knowledge base (shorter).

### Issue: Conversation context lost
**Solution:** Context cached for 30 minutes. Check CONVERSATION_CACHE_EXPIRY setting.

### Issue: API errors appearing in response
**Solution:** Enhanced handler catches errors and falls back to knowledge base. Check logs for details.

### Issue: Same response to different questions
**Solution:** Likely using knowledge base (offline mode). Configure API keys to enable long-form responses.

## Performance Notes

- **API Response Time:** 2-5 seconds (depends on network)
- **Knowledge Base Response:** < 500ms
- **Cache Hit:** Instant (conversation context)
- **Max Cache Size:** ~1GB (adjustable)

## Best Practices

1. **Configure both API keys** — Provides redundancy (Gemini if OpenAI fails)
2. **Monitor API usage** — Both OpenAI and Gemini have rate limits/quotas
3. **Keep API keys secure** — Use `.env` files, never commit keys
4. **Review logs** — Track which response source is being used
5. **Test offline mode** — Ensure knowledge base responses work

## Architecture

```
Chat Message
    ↓
[EnhancedResponseHandler]
├─ [ConversationMemoryCache] → Get context
├─ [Intent Extraction] → Classify question
├─ [Sentiment Analysis] → Detect mood
├─ [API Strategy]
│  ├─ Try OpenAI (if key available)
│  ├─ Fallback to Gemini (if key available)
│  └─ Fallback to Hybrid (knowledge base + NLP)
├─ [History Management]
│  ├─ Add to conversation cache
│  ├─ Track topic progression
│  └─ Maintain metadata
└─ Return response
```

## Security

- API keys loaded from environment variables only
- No keys logged or exposed in errors
- Rate limiting recommended at proxy/LB level
- Cache cleared after 30 minutes (configurable)
