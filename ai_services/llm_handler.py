import os
from openai import OpenAI
from typing import List, Dict, Optional


GENERIC_OPENAI_FALLBACK = "I am temporarily unable to generate an advanced response right now. Please try again in a moment."


def ask_llm(user_input: str, history: List[Dict], system_prompt: Optional[str] = None) -> str:
    """
    Primary Brain (LLM). 
    Handles complex reasoning, coding, and general knowledge.
    """
    try:
        # Load API Key securely
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAPI_API_KEY")
        if not api_key or "sk-" not in api_key:
            return GENERIC_OPENAI_FALLBACK

        client = OpenAI(api_key=api_key)

        # Build message list (System + History + Current)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(history or [])
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})

        # Call OpenAI (Using gpt-4o-mini for speed and cost efficiency)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg:
            return GENERIC_OPENAI_FALLBACK
        if "quota" in error_msg:
            return GENERIC_OPENAI_FALLBACK
        
        print(f"LLM Error: {e}")
        return GENERIC_OPENAI_FALLBACK
