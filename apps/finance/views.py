from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from ai_services.cognitive_ai_integration import CognitiveAIServices


@login_required
def finance_home(request):
    services = CognitiveAIServices(str(request.user.id))
    if hasattr(services, 'get_dashboard_summary'):
        dashboard_summary = services.get_dashboard_summary()
    else:
        dashboard_summary = services.get_dashboard_data()
    anomaly_summary = services.get_anomaly_summary(days=7)
    if not isinstance(anomaly_summary, dict):
        anomaly_summary = {'anomalies': []}
    if not isinstance(dashboard_summary, dict):
        dashboard_summary = {}

    return render(request, 'finance/index.html', {
        'dashboard_summary': dashboard_summary,
        'anomaly_summary': anomaly_summary,
        'finance_cards': [
            {'label': 'Monthly spend', 'value': '₹24,800', 'trend': '+8%'},
            {'label': 'Predicted spend', 'value': '₹26,100', 'trend': 'Forecast'},
            {'label': 'Alerts', 'value': str(len(anomaly_summary.get('anomalies', []))) if isinstance(anomaly_summary, dict) else '0', 'trend': 'Monitor'},
        ],
        'spending_rows': [
            {'category': 'Subscriptions', 'amount': '₹4,200', 'delta': '+4%'},
            {'category': 'Food', 'amount': '₹8,600', 'delta': '+10%'},
            {'category': 'Transport', 'amount': '₹2,150', 'delta': '-3%'},
        ],
    })
