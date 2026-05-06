from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from ai_services.cognitive_ai_integration import CognitiveAIServices
from apps.task_manager.models import Task


@login_required
def planner_home(request):
    services = CognitiveAIServices(str(request.user.id))
    cognitive_state = services.get_cognitive_state()
    tasks = Task.objects.filter(user=request.user).order_by('status', '-priority', 'due_date')[:18]

    task_payload = [
        {
            'id': task.id,
            'title': task.title,
            'priority': task.priority,
            'category': task.category,
            'due_date': task.due_date,
            'status': task.status,
            'estimated_duration': task.estimated_duration,
            'energy_weight': {'high': 3, 'medium': 2, 'low': 1}.get(task.priority, 2),
        }
        for task in tasks
    ]

    recommendations = services.get_task_recommendations(task_payload, cognitive_state)

    return render(request, 'planner/index.html', {
        'tasks': tasks,
        'cognitive_state': cognitive_state,
        'recommendations': recommendations,
        'planner_hours': [8, 10, 12, 14, 16, 18, 20],
    })
