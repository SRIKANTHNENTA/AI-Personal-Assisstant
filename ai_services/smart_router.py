from datetime import datetime
import re
from typing import Optional, Dict

def smart_router(user_input: str, user_memories: Dict) -> Optional[str]:
    """
    High-speed local routing engine. 
    Handles basic conversation, identity, time, and math without API calls.
    """
    text = user_input.lower().strip()

    # ======================
    # 1. GREETINGS
    # ======================
    if text in ["hi", "hello", "hey", "greetings"]:
        return "Hey! How can I help you?"

    # ======================
    # 3. TIME & DATE
    # ======================
    if "time" in text and ("what" in text or "tell" in text):
        return f"The current time is {datetime.now().strftime('%I:%M %p')}."

    if "date" in text or "today" in text:
        return f"Today is {datetime.now().strftime('%A, %d %B %Y')}."

    # ======================
    # 4. SIMPLE MATH
    # ======================
    # Look for patterns like "1+1" or "22 * 5"
    math_match = re.search(r"(\d+)\s*([\+\-\*/])\s*(\d+)", text)
    if math_match:
        try:
            # We use a safe subset or simple calculation here
            num1 = int(math_match.group(1))
            op = math_match.group(2)
            num2 = int(math_match.group(3))
            if op == '+': res = num1 + num2
            elif op == '-': res = num1 - num2
            elif op == '*': res = num1 * num2
            elif op == '/': res = num1 / num2 if num2 != 0 else "undefined"
            return f"The result is {res}."
        except:
            pass

    # ======================
    # 5. SHORT INPUTS (Safety)
    # ======================
    # Only intercept very low-information fillers. Single-word technical queries
    # like "threading" should proceed to the intent/LLM pipeline.
    low_info_tokens = {"hmm", "huh", "uh", "h", "?", "..."}
    if len(text.split()) <= 1 and text in low_info_tokens:
        return "Can you please give more details? 😊"

    # ======================
    # NOTHING MATCHED (Proceed to LLM)
    # ======================
    return None
