# WebSocket Setup Instructions

## Issue: WebSocket Connection Failed

The WebSocket connection is failing because:
1. Redis server is not running (required for Django Channels)
2. The server needs to run with Daphne (ASGI server) instead of the default Django runserver

## Solution: Two Options

### Option 1: Use Chat Without WebSocket (Simple)

The chat interface has a fallback HTTP mode. You can use the chat without WebSocket by:
1. Just use the application as-is
2. Messages will work but won't be real-time
3. You'll need to refresh to see new messages

**This works immediately without any additional setup!**

### Option 2: Enable Real-Time WebSocket Chat (Advanced)

For full real-time chat functionality, follow these steps:

#### Step 1: Install Redis

**Windows:**
1. Download Redis for Windows from: https://github.com/microsoftarchive/redis/releases
2. Or use WSL (Windows Subsystem for Linux):
   ```bash
   wsl --install
   # Then in WSL:
   sudo apt update
   sudo apt install redis-server
   sudo service redis-server start
   ```

**Alternative - Use Docker:**
```bash
docker run -d -p 6379:6379 redis
```

#### Step 2: Verify Redis is Running

```bash
# Test Redis connection
redis-cli ping
# Should return: PONG
```

#### Step 3: Run with Daphne (ASGI Server)

Instead of `python manage.py runserver`, use:

```bash
daphne -b 127.0.0.1 -p 8000 config.asgi:application
```

Or install and use:
```bash
pip install daphne
python -m daphne config.asgi:application
```

#### Step 4: Access the Application

Visit: http://127.0.0.1:8000

Now WebSocket chat will work in real-time!

---

## Current Status

✅ **Application is working** with HTTP fallback
✅ **All features work** except real-time WebSocket chat
⏳ **WebSocket requires** Redis + Daphne server

## Recommendation

**For now:** Use the application as-is. The chat works fine without WebSocket, just not in real-time.

**Later:** When you want real-time features, install Redis and run with Daphne.

---

## Quick Test Without WebSocket

1. Go to: http://127.0.0.1:8000/auth/register/
2. Register a new user
3. Login and go to chat
4. Type messages - they will work via HTTP POST
5. The AI will respond (if OpenAI API key is configured)

The only difference is messages won't appear instantly - you may need to refresh the page.
