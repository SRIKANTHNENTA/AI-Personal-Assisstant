"""
Task Manager Views
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.urls import reverse
from django.utils.dateparse import parse_datetime
from datetime import timedelta
from .models import Task, Reminder, TaskCompletion, PredictiveInsight
from ai_services.behavior_learner import BehaviorLearner
from ai_services.cognitive_ai_integration import CognitiveAIServices


@login_required
def task_list_view(request):
    """List all tasks"""
    status = (request.GET.get('status') or 'all').strip().lower()
    tasks = Task.objects.filter(user=request.user)

    if status in {'pending', 'completed', 'in_progress', 'cancelled', 'overdue'}:
        tasks = tasks.filter(status=status)

    tasks = tasks.order_by('-priority', 'due_date')
    
    # Get insights
    insights = PredictiveInsight.objects.filter(
        user=request.user,
        is_acted_upon=False
    ).order_by('-confidence_score')[:5]
    
    context = {
        'tasks': tasks,
        'insights': insights,
        'selected_status': status,
        'now_iso': timezone.localtime(timezone.now()).isoformat(),
    }
    
    return render(request, 'task_manager/task_list.html', context)


@login_required
@require_http_methods(["GET", "POST"])
def create_task_view(request):
    """Create a new task"""
    categories = [choice[0] for choice in Task.CATEGORY_CHOICES]
    priorities = [choice[0] for choice in Task.PRIORITY_CHOICES]

    if request.method == 'POST':
        title = (request.POST.get('title') or '').strip()
        description = request.POST.get('description', '')
        category = request.POST.get('category', 'other')
        priority = request.POST.get('priority', 'medium')
        due_date_raw = (request.POST.get('due_date') or '').strip()
        estimated_duration = request.POST.get('estimated_duration')
        remind_before_minutes = request.POST.get('remind_before_minutes', '15')

        if not title:
            return render(request, 'task_manager/create_task.html', {
                'error': 'Task title is required.',
                'form_data': request.POST,
                'now_local': timezone.localtime(timezone.now()).strftime('%Y-%m-%dT%H:%M'),
            })

        due_date = None
        if due_date_raw:
            due_date = parse_datetime(due_date_raw)
            if due_date is None:
                # `datetime-local` sometimes arrives without seconds/timezone.
                try:
                    due_date = timezone.datetime.strptime(due_date_raw, '%Y-%m-%dT%H:%M')
                except ValueError:
                    due_date = None

            if due_date is None:
                return render(request, 'task_manager/create_task.html', {
                    'error': 'Invalid deadline format. Please choose a valid date and time.',
                    'form_data': request.POST,
                    'now_local': timezone.localtime(timezone.now()).strftime('%Y-%m-%dT%H:%M'),
                })

            if timezone.is_naive(due_date):
                due_date = timezone.make_aware(due_date, timezone.get_current_timezone())

            if due_date <= timezone.now():
                return render(request, 'task_manager/create_task.html', {
                    'error': 'Deadline must be in the future.',
                    'form_data': request.POST,
                    'now_local': timezone.localtime(timezone.now()).strftime('%Y-%m-%dT%H:%M'),
                })

        if category not in categories:
            category = 'other'
        if priority not in priorities:
            priority = 'medium'

        try:
            estimated_duration_value = int(estimated_duration) if estimated_duration else None
        except ValueError:
            estimated_duration_value = None

        task = Task.objects.create(
            user=request.user,
            title=title,
            description=description,
            category=category,
            priority=priority,
            due_date=due_date,
            estimated_duration=estimated_duration_value,
        )

        # Create reminder if due date is set
        if due_date:
            try:
                lead_minutes = max(1, int(remind_before_minutes))
            except ValueError:
                lead_minutes = 15

            reminder_at = due_date - timedelta(minutes=lead_minutes)

            # If lead-time reminder would be in the past, use a near-immediate reminder.
            if reminder_at <= timezone.now():
                reminder_at = timezone.now() + timedelta(seconds=30)

            Reminder.objects.create(
                task=task,
                user=request.user,
                reminder_time=reminder_at,
                message=f"Upcoming task: {task.title} in {lead_minutes} minute(s)."
            )

            Reminder.objects.create(
                task=task,
                user=request.user,
                reminder_time=due_date,
                message=f"Task time reached: {task.title}"
            )

        return redirect('task_list')

    return render(request, 'task_manager/create_task.html', {
        'now_local': timezone.localtime(timezone.now()).strftime('%Y-%m-%dT%H:%M')
    })


@login_required
@require_http_methods(["POST"])
def complete_task(request, task_id):
    """Mark task as completed"""
    task = get_object_or_404(Task, id=task_id, user=request.user)
    
    task.status = 'completed'
    task.completed_at = timezone.now()
    task.save()
    
    # Calculate completion time
    completion_time = (task.completed_at - task.created_at).total_seconds() / 60
    
    # Create completion record
    TaskCompletion.objects.create(
        user=request.user,
        task=task,
        completion_time=int(completion_time),
        on_time=task.completed_at <= task.due_date if task.due_date else True
    )
    
    return JsonResponse({'success': True, 'message': 'Task completed!'})


@login_required
@require_http_methods(["POST"])
def delete_task(request, task_id):
    """Delete a user task."""
    task = get_object_or_404(Task, id=task_id, user=request.user)
    task.delete()
    return JsonResponse({'success': True})


@login_required
def reminder_feed(request):
    """Return near-due reminders and approaching tasks for browser notifications."""
    now = timezone.now()
    lookahead = now + timedelta(minutes=30)

    reminders = Reminder.objects.filter(
        user=request.user,
        status='pending',
        reminder_time__lte=lookahead,
        reminder_time__gte=now - timedelta(minutes=1),
    ).select_related('task').order_by('reminder_time')[:20]

    reminder_payload = []
    to_mark_sent = []
    for reminder in reminders:
        task = reminder.task
        if task.status in {'completed', 'cancelled'}:
            reminder.status = 'dismissed'
            to_mark_sent.append(reminder)
            continue

        seconds_left = int((reminder.reminder_time - now).total_seconds())
        reminder_payload.append({
            'id': reminder.id,
            'task_id': task.id,
            'task_title': task.title,
            'message': reminder.message or f"Reminder: {task.title}",
            'seconds_left': seconds_left,
            'reminder_time': timezone.localtime(reminder.reminder_time).isoformat(),
        })

        # Mark as sent when reminder window is reached.
        if reminder.reminder_time <= now:
            reminder.status = 'sent'
            reminder.sent_at = now
            to_mark_sent.append(reminder)

    if to_mark_sent:
        Reminder.objects.bulk_update(to_mark_sent, ['status', 'sent_at'])

    approaching_tasks = Task.objects.filter(
        user=request.user,
        status__in=['pending', 'in_progress'],
        due_date__isnull=False,
        due_date__lte=lookahead,
    ).order_by('due_date')[:20]

    task_payload = []
    for task in approaching_tasks:
        seconds_to_due = int((task.due_date - now).total_seconds())
        task_payload.append({
            'id': task.id,
            'title': task.title,
            'priority': task.priority,
            'seconds_to_due': seconds_to_due,
            'due_at': timezone.localtime(task.due_date).isoformat(),
        })

    return JsonResponse({
        'now': timezone.localtime(now).isoformat(),
        'reminders': reminder_payload,
        'approaching_tasks': task_payload,
    })


@login_required
def dashboard_view(request):
    """Main user dashboard"""
    user = request.user
    
    # Get tasks
    pending_tasks = Task.objects.filter(user=user, status='pending').count()
    completed_today = Task.objects.filter(
        user=user,
        status='completed',
        completed_at__date=timezone.now().date()
    ).count()
    
    # Get recent conversations
    from apps.chat_companion.models import Conversation
    recent_conversations = Conversation.objects.filter(user=user).order_by('-last_message_at')[:5]
    
    # Get emotion data
    from apps.emotion_tracker.models import EmotionState
    recent_emotions = EmotionState.objects.filter(user=user).order_by('-detected_at')[:10]
    
    # Get Cognitive AI data
    services = CognitiveAIServices(str(user.id))
    cognitive_state = services.get_cognitive_state()
    
    # Get RL-based task recommendations
    pending_tasks_list = Task.objects.filter(user=user, status='pending')
    tasks_data = []
    for t in pending_tasks_list:
        tasks_data.append({
            'id': t.id,
            'title': t.title,
            'priority': t.priority,
            'category': t.category,
            'estimated_duration': t.estimated_duration
        })
    
    recommendations = services.get_task_recommendations(tasks_data, cognitive_state)

    module_cards = [
        {
            'title': 'AI Chat',
            'description': 'Streaming conversation with memory and context awareness.',
            'icon': 'fa-comments',
            'tag': 'Conversation',
            'href': reverse('chat_companion:chat'),
        },
        {
            'title': 'Voice',
            'description': 'Speech capture, transcription, and voice feedback.',
            'icon': 'fa-microphone-alt',
            'tag': 'Audio',
            'href': reverse('voice_assistant:home'),
        },
        {
            'title': 'Notes',
            'description': 'AI summaries, semantic search, and memory linking.',
            'icon': 'fa-note-sticky',
            'tag': 'Memory',
            'href': reverse('notes:home'),
        },
        {
            'title': 'Scan',
            'description': 'Vision capture, object detection, OCR, and emotion cues.',
            'icon': 'fa-camera-retro',
            'tag': 'Vision',
            'href': reverse('vision:home'),
        },
        {
            'title': 'Music',
            'description': 'Mood-based playback with voice control and visualizer.',
            'icon': 'fa-music',
            'tag': 'Audio AI',
            'href': reverse('music:home'),
        },
        {
            'title': 'Email AI',
            'description': 'Classification, auto-reply suggestions, and prioritization.',
            'icon': 'fa-envelope-open-text',
            'tag': 'Inbox',
            'href': reverse('email_ai:home'),
        },
    ]
    
    context = {
        'pending_tasks': pending_tasks,
        'completed_today': completed_today,
        'recent_conversations': recent_conversations,
        'recent_emotions': recent_emotions,
        'cognitive_state': cognitive_state,
        'recommendations': recommendations,
        'module_cards': module_cards,
    }
    
    return render(request, 'dashboard/user_dashboard.html', context)
