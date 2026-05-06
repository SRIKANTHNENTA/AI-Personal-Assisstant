"""
Neural Network Chatbot Engine
Uses pre-trained Keras model to predict intents and generate responses
"""

import json
import pickle
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import datetime


class NeuralChatbot:
    """
    Neural network-based chatbot using Keras model
    Predicts user intent and generates appropriate responses
    """
    
    def __init__(self):
        self.model = None
        self.words = None
        self.classes = None
        self.intents = None
        self.taxonomy = None
        self.model_loaded = False
        
        # Try to load model
        try:
            self._load_model()
            self._load_taxonomy()
        except Exception as e:
            print(f"Warning: Could not load neural model: {e}")
            print("Neural chatbot will be disabled. Install tensorflow/keras to enable.")
    
    def _load_model(self):
        """Load the trained model and preprocessed data"""
        try:
            model_dir = Path(__file__).parent / 'neural_model'
            model_path_h5 = model_dir / 'mymodel.h5'
            model_path_keras = model_dir / 'mymodel.keras'

            # Prefer native Keras 3 loader for the .keras artifact we train and save.
            keras_load_model = None
            keras_sequential = None
            keras_dense = None
            keras_dropout = None
            keras_input = None
            try:
                from keras.models import Sequential as keras_sequential  # type: ignore
                from keras.models import load_model as keras_load_model  # type: ignore
                from keras.layers import Dense as keras_dense  # type: ignore
                from keras.layers import Dropout as keras_dropout  # type: ignore
                from keras.layers import Input as keras_input  # type: ignore
            except Exception:
                pass

            # Fallback for environments that only expose tf.keras.
            tf_sequential = None
            tf_load_model = None
            tf_dense = None
            tf_dropout = None
            tf_input = None
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential as tf_sequential, load_model as tf_load_model
                from tensorflow.keras.layers import Dense as tf_dense, Dropout as tf_dropout, Input as tf_input
            except Exception:
                tf = None
            
            # ATTEMPT 1: Modern Keras 3 loading (.keras) - Preferred
            if model_path_keras.exists():
                try:
                    if keras_load_model is not None:
                        self.model = keras_load_model(str(model_path_keras), compile=False)
                    elif tf_load_model is not None:
                        self.model = tf_load_model(str(model_path_keras), compile=False)
                    else:
                        raise RuntimeError("No Keras loader available")
                    print("✅ Neural chatbot model loaded successfully (Modern .keras).")
                except Exception as e:
                    print(f"Modern load failed: {e}. Falling back...")

            # ATTEMPT 2: Legacy H5 loading with robust fallback
            if not self.model and model_path_h5.exists():
                try:
                    # Silently try standard load for .h5
                    if keras_load_model is not None:
                        self.model = keras_load_model(str(model_path_h5), compile=False)
                    elif tf_load_model is not None:
                        self.model = tf_load_model(str(model_path_h5), compile=False)
                    else:
                        raise RuntimeError("No Keras loader available")
                    print("✅ Neural chatbot model loaded successfully (Standard H5).")
                except Exception:
                    # Final fallback: use a simple inferred architecture only when needed.
                    try:
                        input_size = 0
                        output_size = 0
                        try:
                            with open(model_dir / 'words.pkl', 'rb') as f:
                                input_size = len(pickle.load(f))
                            with open(model_dir / 'classes.pkl', 'rb') as f:
                                output_size = len(pickle.load(f))
                        except Exception:
                            input_size = 0
                            output_size = 0

                        if not input_size or not output_size:
                            raise RuntimeError("Cannot infer model shape from words/classes")

                        if keras_sequential is not None:
                            self.model = keras_sequential([
                                keras_input(shape=(input_size,)),
                                keras_dense(128, activation='relu'),
                                keras_dropout(0.5),
                                keras_dense(64, activation='relu'),
                                keras_dropout(0.3),
                                keras_dense(output_size, activation='softmax')
                            ])
                        elif tf_sequential is not None:
                            self.model = tf_sequential([
                                tf_input(shape=(input_size,)),
                                tf_dense(128, activation='relu'),
                                tf_dropout(0.5),
                                tf_dense(64, activation='relu'),
                                tf_dropout(0.3),
                                tf_dense(output_size, activation='softmax')
                            ])
                        else:
                            raise RuntimeError("No model builder available")

                        self.model.load_weights(str(model_path_h5))
                        print("✅ Neural chatbot model loaded successfully (Legacy H5 fallback).")
                    except Exception as e2:
                        print(f"❌ Critical Error: Could not reconstruct neural model: {e2}")
                        raise
            
            # Load preprocessed data (words, classes, intents)
            with open(model_dir / 'words.pkl', 'rb') as f:
                self.words = pickle.load(f)
            
            with open(model_dir / 'classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            
            with open(model_dir / 'intents.json', 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            
            self.model_loaded = True
            print("✅ Neural chatbot model and data loaded successfully!")
            
        except ImportError:
            print("⚠️ TensorFlow/Keras not installed. Neural chatbot disabled.")
            print("Install with: pip install tensorflow keras")
            raise
        except Exception as e:
            print(f"Critical error in _load_model: {e}")
            raise
    def _load_taxonomy(self):
        """Load the intent taxonomy data"""
        try:
            taxonomy_path = Path(__file__).parent / 'intent_taxonomy.json'
            if taxonomy_path.exists():
                with open(taxonomy_path, 'r') as f:
                    self.taxonomy = json.load(f)
                print("📊 Intent taxonomy loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load taxonomy: {e}")

    def clean_up_sentence(self, sentence: str) -> List[str]:
        """
        Tokenize and lemmatize the sentence
        """
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            
            # Download required NLTK data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            lemmatizer = WordNetLemmatizer()
            sentence_words = nltk.word_tokenize(sentence)
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            return sentence_words
            
        except Exception as e:
            print(f"Error in sentence cleanup: {e}")
            return sentence.lower().split()
    
    def create_bow(self, sentence: str) -> np.ndarray:
        """
        Create bag of words array from sentence
        """
        sentence_words = self.clean_up_sentence(sentence)
        bag = list(np.zeros(len(self.words)))
        
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        
        return np.array(bag)
    
    def predict_intent(self, message: str, confidence_threshold: float = 0.75) -> Optional[Dict]:
        """
        Predict intent from user message
        Returns dict with intent and confidence, or None if below threshold
        """
        if not self.model_loaded:
            return None
        
        try:
            # Create bag of words
            bow = self.create_bow(message)
            
            # Predict
            res = self.model.predict(np.array([bow]), verbose=0)[0]
            
            # Get results above threshold
            results = [[i, r] for i, r in enumerate(res) if r > confidence_threshold]
            
            if not results:
                return None
            
            # Sort by confidence
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return best match
            intent_index = results[0][0]
            confidence = results[0][1]
            
            return {
                'intent': self.classes[intent_index],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return None
    
    def get_response(self, message: str) -> Optional[str]:
        """
        Get response for user message using neural network
        Returns response string or None if no confident match
        """
        if not self.model_loaded:
            return None
        
        # Predict intent
        prediction = self.predict_intent(message)
        
        if not prediction:
            return None
        
        intent_tag = prediction['intent']
        
        # Handle special intents
        if intent_tag == 'datetime':
            return self._get_datetime_response()
        
        if intent_tag == 'options':
            return self._get_taxonomy_summary()
        
        if intent_tag == 'expertise_query':
            return self._get_specific_expertise(message)
        
        # Find intent in intents.json
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                # Return random response from this intent
                return random.choice(intent['responses'])
        
        return None

    def _get_taxonomy_summary(self) -> str:
        """Generate a summary of capabilities based on taxonomy"""
        if not self.taxonomy:
            return "I am a general purpose AI assistant. I can help with programming, career guidance, and daily tasks."
        
        top_categories = [item['category'] for item in self.taxonomy['intent_taxonomy'] if item['count'] >= 100]
        
        if not top_categories:
            return "I am trained across various domains including tech, life, and science."
            
        summary = "I am a high-capacity AI. My strongest areas of training are in " + ", ".join(top_categories) + ". "
        summary += "I also have specialized modules for over 30 other categories like AI/ML, Travel, and Finance. How can I assist you today?"
        return summary

    def _get_specific_expertise(self, message: str) -> str:
        """Fetch specific intent counts for mentioned categories"""
        if not self.taxonomy:
            return "My expertise database is currently being updated. I am generally trained in most technical and personal topics."
            
        message_lower = message.lower()
        for item in self.taxonomy['intent_taxonomy']:
            category = item['category']
            if category.lower() in message_lower:
                return f"I have {item['count']} specialized intents for {category}. I am highly optimized to provide detailed responses in this domain."
                
        return "I'm not specifically trained in that exact sub-category yet, but I have broad general knowledge that can help! What's on your mind?"
    
    def _get_datetime_response(self) -> str:
        """Get current date and time"""
        now = datetime.datetime.now()
        return f"📅 {now.strftime('%A, %B %d, %Y')}\n🕐 {now.strftime('%I:%M %p')}"
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {
                'loaded': False,
                'error': 'Model not loaded'
            }
        
        return {
            'loaded': True,
            'vocabulary_size': len(self.words),
            'num_intents': len(self.classes),
            'intents': self.classes
        }
