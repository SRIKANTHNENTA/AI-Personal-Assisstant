from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib.auth import get_user_model

from ai_services.cognitive_ai_integration import CognitiveAIServices
from ai_services.nlp_engine import NLPEngine

User = get_user_model()

# More realistic sample emails with varied topics and tones
SAMPLE_EMAILS = [
    {
        'sender': 'manager@company.com',
        'sender_name': 'Sarah Chen',
        'subject': 'URGENT: Project deadline moved up by 2 weeks',
        'body': 'Hi, the client just changed the delivery date to next Friday. We need to accelerate the development timeline. Please prioritize this and update the team. Can you confirm if this is feasible?',
        'timestamp': '2 hours ago',
    },
    {
        'sender': 'support@newsletter.com',
        'sender_name': 'Newsletter Bot',
        'subject': 'Unsubscribe from weekly updates?',
        'body': 'You are receiving this email because you subscribed. Click here to unsubscribe or manage your preferences.',
        'timestamp': '4 hours ago',
    },
    {
        'sender': 'john.review@team.dev',
        'sender_name': 'John Parker',
        'subject': 'Code review feedback on PR #245',
        'body': 'I reviewed your pull request. Great work on the optimization! I have a few minor suggestions about error handling. Can we sync up tomorrow to discuss?',
        'timestamp': '6 hours ago',
    },
    {
        'sender': 'billing@service.com',
        'sender_name': 'Service Billing',
        'subject': 'Your subscription renewal is coming up',
        'body': 'Your annual subscription will renew on April 15, 2026 for $299. No action needed unless you want to make changes.',
        'timestamp': '1 day ago',
    },
    {
        'sender': 'team@marketing.co',
        'sender_name': 'Marketing Team',
        'subject': 'Join us for Q2 planning session',
        'body': 'We are organizing a virtual planning session next Tuesday at 2pm. Please confirm your attendance. Agenda includes roadmap review and resource allocation.',
        'timestamp': '1 day ago',
    },
    {
        'sender': 'hr@company.com',
        'sender_name': 'HR Department',
        'subject': 'ACTION REQUIRED: Complete your annual survey',
        'body': 'Please complete your annual employee survey by Friday. Your feedback is valuable and will take approximately 10 minutes.',
        'timestamp': '2 days ago',
    },
    {
        'sender': 'alerts@monitoring.io',
        'sender_name': 'System Alerts',
        'subject': 'CRITICAL: Server CPU usage at 95%',
        'body': 'Alert triggered at 2026-04-05 14:32:10 UTC. Your production server is experiencing high CPU load. Recommended actions: check running processes or scale up.',
        'timestamp': '30 minutes ago',
    },
    {
        'sender': 'client@enterprise.biz',
        'sender_name': 'Michael Torres',
        'subject': 'Following up on proposal we discussed',
        'body': 'Hi, I wanted to follow up on the proposal I sent last week. Do you have any questions or are we ready to move forward? Looking forward to your thoughts.',
        'timestamp': '3 days ago',
    },
    {
        'sender': 'noreply@shipping.com',
        'sender_name': 'Shipping Notifications',
        'subject': 'Your package is out for delivery',
        'body': 'Your order #ORD-2026-98765 is out for delivery today. Estimated arrival: 3-5 PM. Track your package for more details.',
        'timestamp': '4 hours ago',
    },
    {
        'sender': 'colleague@team.dev',
        'sender_name': 'Emma Wilson',
        'subject': 'Can we grab coffee and chat about the new project?',
        'body': 'Hey! I wanted to pick your brain about the new initiative we discussed. Are you free for coffee this week? Would love to hear your ideas.',
        'timestamp': '5 hours ago',
    },
    {
        'sender': 'payment-failed@shop.com',
        'sender_name': 'Payment System',
        'subject': 'Payment failed - action required',
        'body': 'Your recent payment attempt failed. Please update your payment method to continue service. Click here to manage billing.',
        'timestamp': '30 minutes ago',
    },
    {
        'sender': 'docs@platform.io',
        'sender_name': 'Documentation Bot',
        'subject': 'New API documentation released',
        'body': 'We have released updated documentation for API v2.3. Check out the new features and breaking changes in the release notes.',
        'timestamp': '1 day ago',
    },
]


class EmailClassifier:
    """Classifies emails using NLP and keyword analysis."""
    
    URGENT_KEYWORDS = ['urgent', 'critical', 'action required', 'asap', 'immediately', 'deadline', 'emergency', 'fail', 'error', 'alert', 'moved up']
    FOLLOW_UP_KEYWORDS = ['follow up', 'following up', 'as discussed', 'proposal', 'confirm', 'sync', 'touch base', 'checking in']
    NOTIFICATION_KEYWORDS = ['confirm', 'renew', 'subscription', 'alert', 'notification', 'tracking', 'delivery', 'shipped', 'status']
    SPAM_KEYWORDS = ['unsubscribe', 'prefers', 'noreply', 'newsletter', 'received this because you subscribed']
    
    def __init__(self):
        self.nlp_engine = NLPEngine()
    
    def classify(self, email: dict) -> dict:
        """Classify email and determine category and priority."""
        subject = (email.get('subject') or '').lower()
        body = (email.get('body') or '').lower()
        combined_text = f"{subject} {body}"
        
        # Check for spam first
        if any(keyword in combined_text for keyword in self.SPAM_KEYWORDS):
            return {'category': 'spam', 'priority': 'low', 'urgency_score': 0.1}
        
        # Check for urgent emails
        if any(keyword in combined_text for keyword in self.URGENT_KEYWORDS):
            return {'category': 'urgent', 'priority': 'high', 'urgency_score': 0.95}
        
        # Check for follow-ups
        if any(keyword in combined_text for keyword in self.FOLLOW_UP_KEYWORDS):
            return {'category': 'follow-up', 'priority': 'medium', 'urgency_score': 0.65}
        
        # Check for notifications
        if any(keyword in combined_text for keyword in self.NOTIFICATION_KEYWORDS):
            return {'category': 'notification', 'priority': 'low', 'urgency_score': 0.35}
        
        # Analyze sentiment for general priority
        try:
            sentiment = self.nlp_engine.analyze_sentiment(combined_text)
            compound = sentiment.get('compound', 0.0)
            
            if compound < -0.3:
                return {'category': 'concern', 'priority': 'high', 'urgency_score': 0.80}
            elif compound > 0.3:
                return {'category': 'positive', 'priority': 'medium', 'urgency_score': 0.50}
        except Exception:
            pass
        
        # Default to personal/general
        return {'category': 'personal', 'priority': 'medium', 'urgency_score': 0.55}


def _generate_ai_reply(sender_name: str, subject: str, body: str, classification: dict, services: CognitiveAIServices) -> str:
    """Generate an AI-powered reply suggestion based on email content and classification."""
    category = classification.get('category', 'general')
    
    # Build context for the AI
    context_prompts = {
        'urgent': f"Generate a professional, concise reply acknowledging urgency and committing to action quickly.",
        'follow-up': f"Generate a professional reply that shows engagement and moves the conversation forward.",
        'notification': f"Generate a brief acknowledgment reply that confirms receipt and thanks them.",
        'spam': f"N/A - This appears to be promotional content.",
        'concern': f"Generate an empathetic, solution-focused reply that addresses the concern.",
        'positive': f"Generate a warm, appreciative reply that builds on the positive tone.",
        'personal': f"Generate a friendly, professional reply that matches the tone of the email.",
    }
    
    if category == 'spam':
        return "Consider unsubscribing or marking as spam."
    
    prompt = context_prompts.get(category, "Generate a professional reply.")
    
    try:
        reply = services.generate_response(
            user_message=f"Draft a reply to this email from {sender_name} (Subject: {subject}). {prompt} Keep it to 2-3 sentences.",
            conversation_history=[]
        )
        return reply.strip() if reply else f"Thank you for your email. I will respond shortly."
    except Exception as e:
        print(f"Error generating AI reply: {e}")
        return f"Thank you for reaching out. I will get back to you shortly."


@login_required
def email_home(request):
    query = request.GET.get('q', '').strip().lower()
    
    # Classify and enrich all emails
    classifier = EmailClassifier()
    enriched_emails = []
    
    for email in SAMPLE_EMAILS:
        classification = classifier.classify(email)
        
        enriched = {
            **email,
            'category': classification['category'],
            'priority': classification['priority'],
            'urgency_score': classification['urgency_score'],
        }
        enriched_emails.append(enriched)
    
    # Sort by urgency score (descending)
    enriched_emails.sort(key=lambda x: x['urgency_score'], reverse=True)
    
    # Apply search filter
    inbox = enriched_emails
    if query:
        inbox = [
            item for item in enriched_emails
            if query in item['sender_name'].lower()
            or query in item['subject'].lower()
            or query in item['body'].lower()
            or query in item['category'].lower()
        ]
    
    # Generate AI replies for displayed emails
    try:
        services = CognitiveAIServices(str(request.user.id))
        for item in inbox[:5]:  # Generate replies for top 5
            item['ai_reply'] = _generate_ai_reply(
                item['sender_name'],
                item['subject'],
                item['body'],
                {'category': item['category']},
                services
            )
    except Exception as e:
        print(f"Error generating AI replies: {e}")
        for item in inbox:
            item['ai_reply'] = "Thank you for your email. I will respond shortly."
    
    # Calculate statistics
    stats = {
        'total': len(enriched_emails),
        'urgent': len([e for e in enriched_emails if e['priority'] == 'high']),
        'follow_ups': len([e for e in enriched_emails if e['category'] == 'follow-up']),
        'needs_reply': len([e for e in enriched_emails if e['category'] not in ['notification', 'spam']]),
    }

    return render(request, 'email_ai/index.html', {
        'query': query,
        'inbox': inbox,
        'stats': stats,
    })
