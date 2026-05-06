from django.apps import AppConfig
import os


class ChatCompanionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.chat_companion'

    def ready(self):
        """Initialize AI components on startup"""
        import threading
        import sys
        
        # Don't run twice during auto-reload
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            return
            
        def preload_ai():
            try:
                print("🧠 Initializing AI components...")
                
                # Pre-load Multimodal models
                from ai_services.multimodal_emotion_detector import MultimodalEmotionDetector
                from ai_services.cognitive_ai_integration import CognitiveAIServices
                detector = MultimodalEmotionDetector()
                detector.preload_all_models()

                # Reuse this prewarmed detector across all CognitiveAIServices instances.
                CognitiveAIServices._shared_emotion_detector = detector
                
                # Pre-load Neural Chatbot
                from ai_services.neural_chatbot import NeuralChatbot
                _ = NeuralChatbot()
                
                print("🚀 AI Ready!")
            except Exception as e:
                print(f"❌ Error during AI pre-loading: {e}")

        # Run in thread to not block server startup
        threading.Thread(target=preload_ai, daemon=True).start()
