import os
import google.generativeai as genai
from typing import List, Dict, Optional


GENERIC_GEMINI_FALLBACK = "I am temporarily unable to generate an advanced response right now. Please try again in a moment."


def ask_gemini(user_input: str, history: List[Dict], system_prompt: Optional[str] = None) -> str:
    """
    Google Gemini Brain. 
    Handles complex reasoning, coding, and general help as a replacement for OpenAI.
    """
    try:
        # Load API Key securely
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or "AIza" not in api_key:
            return GENERIC_GEMINI_FALLBACK

        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Model can be configured via env; defaults to a fast general-purpose model.
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )

        # Convert standard chat history to Gemini format
        # Standard: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        # Gemini: [{'role': 'user', 'parts': ['...']}, {'role': 'model', 'parts': ['...']}]
        gemini_history = []
        for msg in (history or []):
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [msg['content']]})

        # Start chat session
        chat = model.start_chat(history=gemini_history)
        
        # Send message
        response = chat.send_message(user_input)

        return response.text

    except Exception as e:
        error_msg = str(e).lower()
        if "api_key_invalid" in error_msg:
            return GENERIC_GEMINI_FALLBACK
        if "quota" in error_msg:
            return GENERIC_GEMINI_FALLBACK
        
        print(f"Gemini Error: {e}")
        return GENERIC_GEMINI_FALLBACK
