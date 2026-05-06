"""
Speech Processing Module
Handles Speech-to-Text and Text-to-Speech functionality
"""

import os
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
from typing import Dict, Optional
from django.conf import settings


class SpeechProcessor:
    """Handles speech recognition and synthesis"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Better defaults for noisy real-world microphones.
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 0.8
        self.recognizer.non_speaking_duration = 0.5
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
    
    def _configure_tts(self):
        """Configure text-to-speech engine"""
        try:
            # Set properties
            self.tts_engine.setProperty('rate', 150)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Use first available voice (can be customized)
                self.tts_engine.setProperty('voice', voices[0].id)
        except Exception as e:
            print(f"TTS configuration error: {e}")
    
    def speech_to_text(self, audio_file_path: str = None, language: str = 'en-US') -> Dict[str, any]:
        """
        Convert speech to text
        If audio_file_path is None, uses microphone input
        """
        try:
            if audio_file_path:
                # Process audio file
                with sr.AudioFile(audio_file_path) as source:
                    audio = self.recognizer.record(source)
            else:
                # Use microphone
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Prefer cloud recognizer with explicit credentials when available.
            text = None
            gcloud_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
            if gcloud_creds:
                try:
                    text = self.recognizer.recognize_google_cloud(audio, language=language)
                except Exception:
                    text = None

            # Fallback to public Google recognizer if cloud path is unavailable.
            if not text:
                text = self.recognizer.recognize_google(audio, language=language)
            
            return {
                'success': True,
                'text': text,
                'confidence': 0.9,  # Google API doesn't return confidence
                'language': language
            }
            
        except sr.UnknownValueError:
            return {
                'success': False,
                'text': '',
                'error': 'Could not understand audio',
                'confidence': 0.0
            }
        except sr.RequestError as e:
            return {
                'success': False,
                'text': '',
                'error': f'Speech recognition service error: {e}',
                'confidence': 0.0
            }
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'error': f'Error: {str(e)}',
                'confidence': 0.0
            }
    
    def text_to_speech(self, text: str, output_file: str = None, language: str = 'en') -> Dict[str, any]:
        """
        Convert text to speech
        If output_file is provided, saves to file. Otherwise, plays directly.
        """
        try:
            if output_file:
                # Use gTTS for file output (better quality)
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(output_file)
                
                return {
                    'success': True,
                    'output_file': output_file,
                    'method': 'gTTS'
                }
            else:
                # Use pyttsx3 for direct playback
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
                return {
                    'success': True,
                    'method': 'pyttsx3'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'TTS error: {str(e)}'
            }
    
    def analyze_voice_features(self, audio_file_path: str) -> Dict[str, any]:
        """
        Analyze voice features for emotion detection
        This is a placeholder - implement with librosa for better results
        """
        try:
            # Placeholder implementation
            # In production, use librosa to extract:
            # - Pitch (fundamental frequency)
            # - Energy/Volume
            # - Speech rate
            # - Spectral features
            
            return {
                'success': True,
                'features': {
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'energy_mean': 0.0,
                    'speech_rate': 0.0,
                },
                'note': 'Placeholder - implement with librosa for real analysis'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Voice analysis error: {str(e)}'
            }


class VoiceCommandProcessor:
    """Process voice commands and extract actions"""
    
    def __init__(self):
        self.speech_processor = SpeechProcessor()
        self.command_keywords = {
            'create_task': ['create task', 'add task', 'new task', 'remind me to'],
            'list_tasks': ['show tasks', 'list tasks', 'what are my tasks'],
            'complete_task': ['complete task', 'finish task', 'done with'],
            'check_emotion': ['how am i feeling', 'my mood', 'emotional state'],
            'help': ['help', 'what can you do'],
        }
    
    def process_voice_command(self, audio_file_path: str = None) -> Dict[str, any]:
        """
        Process voice input and extract command
        """
        # Convert speech to text
        stt_result = self.speech_processor.speech_to_text(audio_file_path)
        
        if not stt_result['success']:
            return {
                'success': False,
                'error': stt_result.get('error', 'Speech recognition failed')
            }
        
        text = stt_result['text'].lower()
        
        # Detect command
        detected_command = 'general_conversation'
        for command, keywords in self.command_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_command = command
                break
        
        return {
            'success': True,
            'text': stt_result['text'],
            'command': detected_command,
            'confidence': stt_result['confidence']
        }
