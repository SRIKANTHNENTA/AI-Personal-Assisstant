# AI Personal Assistant

A comprehensive real-time web-based AI Personal Assistant application built with Django that acts as an intelligent companion capable of learning from users, adapting to their routines, detecting emotions, and managing daily tasks.

## 🌟 Features

### AI & Machine Learning
- **Advanced Cognitive Fusion**: Attention-based multi-modal fusion of emotional and cognitive insights from text, voice, facial expressions, and behavior.
- **RL-Driven Scheduling**: Reinforcement Learning (Q-learning and Policy Gradient) optimized task scheduling that learns from user productivity patterns.
- **Explainable AI (XAI)**: Full transparency using SHAP and LIME to explain why the AI made specific scheduling or emotional detection decisions.
- **Natural Language Processing**: OpenAI GPT-4 integration combined with a custom Neural Dialogue Engine for intelligent, context-aware conversations.
- **Sentiment & Emotion Analysis**: Real-time emotion detection using VADER, DeepFace (Facial), and prosody analysis (Voice).
- **Behavioral Modeling**: ML-based habit formation prediction and routine anomaly detection.
- **Medical Assistant Capabilities**: Integrated health-related knowledge for common ailments, symptoms, and care tips with AI-driven disclaimers.

### Multi-Modal Input
- **Text Chat**: Real-time WebSocket-based interface with a memory-persistent dialogue engine.
- **Voice System**: Speech-to-text (Google STT) and emotion-aware prosody analysis.
- **Camera Input**: DeepFace and OpenCV for real-time facial expression and cognitive state monitoring.
- **Multilingual Support**: Automatic detection and translation for global accessibility.

### Task & Productivity Management
- **Smart RL Scheduler**: AI-suggested tasks with priority and optimal time-slot prediction based on energy levels.
- **Cognitive Monitoring**: Real-time tracking of focus, stress, and flow states to suggest "deep work" vs "routine" tasks.
- **Intelligent Reminders**: Context-aware alerts that adapt to user's real-time availability.
- **Productivity Analytics**: Detailed correlation between emotional states and task completion performance.

## 🛠️ Technology Stack

### Backend
- Django 5.0+
- Django Channels (WebSockets)
- Django REST Framework
- Celery (Background Tasks)
- Redis (Caching & Message Broker)

### AI/ML/DL
- OpenAI GPT-4 & Custom Neural Models
- TensorFlow/PyTorch (Policy Gradients)
- scikit-learn & NumPy (Q-Learning)
- SHAP & LIME (Explainable AI)
- transformers (Hugging Face)
- DeepFace & OpenCV
- TextBlob, VADER, NLTK, spaCy

### Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5
- Font Awesome
- Google Fonts (Inter)

### Database
- PostgreSQL (Production)
- SQLite (Development)

## 📊 Project Resources & Configuration

### API Keys & Services
The application integrates several external AI services. Configuration is handled via environment variables in the `.env` file:
- **OpenAI API**: Core LLM capabilities for complex reasoning and conversation.
- **Google Cloud Platform**: Enhanced Speech-to-Text and translation services.
- **Azure AI Services**: Multi-modal facial emotion detection (optional).
- **SMTP Gateway**: Handles task reminders and user notifications via email.

### Intelligence Datasets
The assistant's "brain" is built upon several specialized datasets:
- **Neural Intent Database**: Over 500+ patterns for technical, medical, and lifestyle queries.
- **Cognitive State Map**: Behavioral sequence data for habit and routine prediction.
- **Medical Knowledge Base**: Curated information on symptoms, first aid, and hygiene.
- **Expertise Taxonomy**: Structured hierarchy of the assistant's learned technical domains.

### Integrated Models
- **Keras Intent Classifier**: Dense neural network for rapid intent recognition.
- **Reinforcement Learning Policy**: State-action maps for optimized task scheduling.
- **Cognitive Fusion Engine**: Attention-weighted model for multi-modal state estimation.

## 📋 Prerequisites

- Python 3.10+
- Redis Server
- PostgreSQL (optional, for production)
- OpenAI API Key
- Google Cloud API credentials (optional, for enhanced features)

## 🚀 Installation

### 1. Clone the Repository
```bash
cd personal_assistance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
SECRET_KEY=your-secret-key
DEBUG=True
OPENAI_API_KEY=your-openai-api-key
# Add other API keys as needed
```

### 4. Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser
```bash
python manage.py createsuperuser
```

### 6. Collect Static Files
```bash
python manage.py collectstatic
```

### 7. Run Redis Server
```bash
redis-server
```

### 8. Run Celery Worker (separate terminal)
```bash
celery -A config worker -l info
```

### 9. Run Celery Beat (separate terminal)
```bash
celery -A config beat -l info
```

### 10. Run Development Server
```bash
python manage.py runserver
```

Visit `http://localhost:8000` in your browser.

## 📁 Project Structure

```
personal_assistance/
├── apps/
│   ├── authentication/      # User auth & profiles
│   ├── chat_companion/      # AI chat interface
│   ├── task_manager/        # Task & reminder system
│   ├── emotion_tracker/     # Emotion detection
│   └── admin_dashboard/     # Admin panel
├── ai_services/
│   ├── nlp_engine.py       # NLP processing
│   ├── speech_processor.py # STT & TTS
│   ├── emotion_analyzer.py # Emotion detection
│   ├── translator.py       # Multilingual support
│   └── behavior_learner.py # ML behavior analysis
├── config/                  # Django settings
├── templates/               # HTML templates
├── static/                  # CSS, JS, images
└── media/                   # User uploads
```

## 🎯 Usage

### Register & Login
1. Navigate to the home page
2. Click "Get Started" to register
3. Login with your credentials

### Chat with AI
1. Go to Dashboard
2. Click "Start Chat"
3. Type or use voice input
4. AI responds with emotion-aware messages

### Manage Tasks
1. Click "New Task" from dashboard
2. Set title, priority, and due date
3. Receive smart reminders
4. Track completion analytics

### View Analytics
- Dashboard shows real-time stats
- Emotion timeline tracking
- Task completion rates
- Behavioral patterns

## 🔧 Configuration

### OpenAI API
Required for conversational AI:
```python
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4
```

### Google Cloud APIs (Optional)
For enhanced speech and translation:
```python
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
```

### Celery Settings
For background tasks and reminders:
```python
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## 🧪 Testing

Run tests:
```bash
python manage.py test
```

## 📊 Admin Panel

Access admin panel at `http://localhost:8000/admin/`

Features:
- User management
- Task monitoring
- Emotion analytics
- API usage tracking
- System logs

## 🔒 Security

- CSRF protection enabled
- Secure password hashing
- API key management via environment variables
- User data encryption
- Rate limiting on API endpoints

## 🌐 Deployment

### Production Checklist
1. Set `DEBUG=False`
2. Configure PostgreSQL database
3. Set up proper `ALLOWED_HOSTS`
4. Use environment variables for secrets
5. Configure HTTPS
6. Set up static file serving (CDN)
7. Configure Celery workers
8. Set up monitoring and logging

## 🤝 Contributing

This is a comprehensive AI assistant project. Contributions are welcome!

## 📝 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Django community
- Bootstrap team
- All open-source contributors


