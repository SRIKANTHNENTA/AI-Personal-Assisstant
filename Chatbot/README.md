# 🧠 Neural Dialogue Engine (Ted)

## Overview
This directory houses the **Neural Dialogue Engine**, internally codenamed **Ted**. It serves as the primary intent recognition and conversational core for the AI Personal Assistant. Using Deep Learning and NLP, it classifies user messages and facilitates intelligent interactions.

## 🚀 Key Functionalities

### 1. Intent Classification
- **Neural Core**: Uses a Keras-based Deep Neural Network to classify user queries into over 30+ specialized categories.
- **Pattern Matching**: Robust recognition of greetings, farewells, and general inquiries.

### 2. Specialized Knowledge Domains
- **Medical Assistant**: Provides care tips and common medication info for fever, headaches, cough, and general hygiene (with built-in medical disclaimers).
- **Technical Expertise**: Integrated knowledge of Python basics, HTML structure, and Data Structures.
- **Career Guidance**: Practical tips for job interviews and common HR/Technical questions.
- **General Knowledge**: Information on monuments, history, and real-time capable hooks for news and weather.

### 3. Integration with Django
Ted is seamlessly integrated into the main Django application's `chat_companion` app via the `ai_services` layer, enabling:
- WebSocket-based real-time communication.
- Persistent user memory and fact-learning.
- Context-aware responses based on user history.

## 📁 Directory Structure
- `chatbot_codes/`: Contains the training scripts (`train_chatbot.py`), model (`mymodel.h5`), and the extensive intent database (`intents.json`).
- `UI/`: Legacy standalone UI assets.

## 🛠️ Standalone Usage (Local)
While primary usage is through the Django dashboard, you can run the standalone flask version:
1. Navigate to this directory: `cd Chatbot`
2. Install requirements: `pip install -r requirements.txt`
3. Run the Flask app:
```bash
set FLASK_APP=chatbot_codes/full_code.py
flask run
```

---
*Note: This module is a vital component of the AI Personal Assistant ecosystem.*

