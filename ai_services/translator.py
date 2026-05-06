"""
Multilingual Translator
Handles language detection and translation
"""

import asyncio
import inspect
from typing import Dict, Optional
from langdetect import detect, DetectorFactory

# Try to import googletrans, but handle import errors gracefully
try:
    from googletrans import Translator
except Exception:
    Translator = None

# Set seed for consistent detection
DetectorFactory.seed = 0


class MultilingualTranslator:
    """Handles translation and language detection"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'hi': 'Hindi',
            'zh-cn': 'Chinese (Simplified)',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'ru': 'Russian',
            'pt': 'Portuguese'
        }

    def _resolve_maybe_async(self, value):
        """Resolve values that may be returned as coroutine objects."""
        if inspect.isawaitable(value):
            try:
                return asyncio.run(value)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(value)
                finally:
                    loop.close()
        return value
    
    def detect_language(self, text: str) -> Dict[str, any]:
        """
        Detect language of text
        """
        try:
            normalized = (text or '').strip()
            if len(normalized) < 4:
                return {
                    'success': True,
                    'language_code': 'en',
                    'language_name': 'English',
                    'confidence': 0.6
                }

            # Prefer googletrans detect when available in runtime.
            if Translator is not None:
                try:
                    detected = self._resolve_maybe_async(Translator().detect(normalized))
                    if detected and getattr(detected, 'lang', None):
                        lang_code = detected.lang
                        lang_name = self.supported_languages.get(lang_code, 'Unknown')
                        confidence = float(getattr(detected, 'confidence', 0.9) or 0.9)
                        return {
                            'success': True,
                            'language_code': lang_code,
                            'language_name': lang_name,
                            'confidence': confidence
                        }
                except Exception:
                    pass

            lang_code = detect(text)
            lang_name = self.supported_languages.get(lang_code, 'Unknown')
            
            return {
                'success': True,
                'language_code': lang_code,
                'language_name': lang_name,
                'confidence': 0.9
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language_code': 'en',
                'language_name': 'English',
                'confidence': 0.0
            }
    
    def translate_text(self, text: str, target_lang: str = 'en', source_lang: str = 'auto') -> Dict[str, any]:
        """
        Translate text to target language
        """
        if Translator is None:
            return {
                'success': False,
                'error': 'Translation service unavailable',
                'original_text': text,
                'translated_text': text,
                'source_language': source_lang,
                'target_language': target_lang
            }
        
        try:
            translation = self._resolve_maybe_async(Translator().translate(
                text,
                src=source_lang,
                dest=target_lang
            ))

            translated_text = getattr(translation, 'text', text)
            source_language = getattr(translation, 'src', source_lang)
            
            return {
                'success': True,
                'original_text': text,
                'translated_text': translated_text,
                'source_language': source_language,
                'target_language': target_lang,
                'confidence': 0.9
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_text': text,
                'translated_text': text,
                'source_language': source_lang,
                'target_language': target_lang
            }
    
    def auto_translate_if_needed(self, text: str, user_preferred_lang: str = 'en') -> Dict[str, any]:
        """
        Automatically translate if text is not in user's preferred language
        """
        # Detect language
        detection = self.detect_language(text)
        
        if not detection['success']:
            return {
                'success': False,
                'error': detection.get('error'),
                'text': text,
                'translated': False
            }
        
        detected_lang = detection['language_code']
        
        # If already in preferred language, no translation needed
        if detected_lang == user_preferred_lang:
            return {
                'success': True,
                'text': text,
                'translated': False,
                'language': detected_lang
            }
        
        # Translate to preferred language
        translation = self.translate_text(text, target_lang=user_preferred_lang, source_lang=detected_lang)
        
        if translation['success']:
            return {
                'success': True,
                'text': translation['translated_text'],
                'translated': True,
                'original_text': text,
                'source_language': detected_lang,
                'target_language': user_preferred_lang
            }
        else:
            return {
                'success': False,
                'error': translation.get('error'),
                'text': text,
                'translated': False
            }
