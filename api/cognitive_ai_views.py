"""
Cognitive AI API Views
Django REST Framework views for exposing cognitive AI services.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
import json

from ai_services.cognitive_ai_integration import CognitiveAIServices, APIResponseFormatter, create_ai_services


class BaseAIView(APIView):
    """Base view for AI endpoints"""
    permission_classes = [IsAuthenticated]
    
    def get_ai_services(self, request) -> CognitiveAIServices:
        """Get or create AI services for the current user"""
        user_id = str(request.user.id)

        return create_ai_services(user_id)


class EmotionAnalysisView(BaseAIView):
    """
    POST /api/ai/emotion/analyze/
    Analyze emotion from text, audio, or image.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            text = request.data.get('text')
            # audio_data and image_data would need proper handling
            
            if not text:
                return Response(
                    APIResponseFormatter.error('Text input required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.analyze_emotion(text=text)
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CognitiveStateView(BaseAIView):
    """
    GET /api/ai/cognitive-state/
    Get current cognitive state.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_cognitive_state()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ActivityRecordView(BaseAIView):
    """
    POST /api/ai/activity/record/
    Record user activity for behavior modeling.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            activity_type = request.data.get('activity_type')
            category = request.data.get('category', 'general')
            duration = request.data.get('duration')
            metadata = request.data.get('metadata', {})
            
            if not activity_type:
                return Response(
                    APIResponseFormatter.error('activity_type required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.record_activity(
                activity_type=activity_type,
                category=category,
                duration=duration,
                metadata=metadata
            )
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BehaviorPatternsView(BaseAIView):
    """
    GET /api/ai/behavior/patterns/
    Get detected behavior patterns.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_behavior_patterns()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TaskRecommendationsView(BaseAIView):
    """
    POST /api/ai/tasks/recommend/
    Get AI task scheduling recommendations.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            tasks = request.data.get('tasks', [])
            current_state = request.data.get('current_state')
            
            if not tasks:
                return Response(
                    APIResponseFormatter.error('tasks list required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.get_task_recommendations(
                tasks=tasks,
                current_state=current_state
            )
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TaskCompletionView(BaseAIView):
    """
    POST /api/ai/tasks/complete/
    Record task completion for learning.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            task_id = request.data.get('task_id')
            task = request.data.get('task', {})
            completion_quality = request.data.get('completion_quality', 0.8)
            actual_duration = request.data.get('actual_duration')
            
            if not task_id:
                return Response(
                    APIResponseFormatter.error('task_id required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.record_task_completion(
                task_id=task_id,
                task=task,
                completion_quality=completion_quality,
                actual_duration=actual_duration
            )
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HabitCreateView(BaseAIView):
    """
    POST /api/ai/habits/create/
    Create a new habit to track.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            habit_name = request.data.get('habit_name')
            category = request.data.get('category', 'general')
            frequency = request.data.get('frequency', 'daily')
            difficulty = request.data.get('difficulty', 0.5)
            cue = request.data.get('cue')
            reward = request.data.get('reward')
            
            if not habit_name:
                return Response(
                    APIResponseFormatter.error('habit_name required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.create_habit(
                habit_name=habit_name,
                category=category,
                frequency=frequency,
                difficulty=difficulty,
                cue=cue,
                reward=reward
            )
            return Response(APIResponseFormatter.success(result), status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HabitLogView(BaseAIView):
    """
    POST /api/ai/habits/log/
    Log habit completion.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            habit_id = request.data.get('habit_id')
            completed = request.data.get('completed', True)
            difficulty_felt = request.data.get('difficulty_felt')
            
            if not habit_id:
                return Response(
                    APIResponseFormatter.error('habit_id required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.log_habit_completion(
                habit_id=habit_id,
                completed=completed,
                difficulty_felt=difficulty_felt
            )
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DailyHabitsView(BaseAIView):
    """
    GET /api/ai/habits/today/
    Get today's habit schedule.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_daily_habits()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ConversationProcessView(BaseAIView):
    """
    POST /api/ai/conversation/process/
    Process conversation for personality learning.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            user_message = request.data.get('user_message')
            assistant_response = request.data.get('assistant_response')
            
            if not user_message or not assistant_response:
                return Response(
                    APIResponseFormatter.error('Both user_message and assistant_response required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.process_conversation(
                user_message=user_message,
                assistant_response=assistant_response
            )
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ResponseStyleView(BaseAIView):
    """
    POST /api/ai/personality/style/
    Get recommended response style.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            context = request.data.get('context', {})
            result = services.get_response_style(context)
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SystemPromptView(BaseAIView):
    """
    POST /api/ai/personality/prompt/
    Get LLM system prompt.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            context = request.data.get('context', {})
            result = services.get_system_prompt(context)
            return Response(APIResponseFormatter.success({'prompt': result}))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DashboardView(BaseAIView):
    """
    GET /api/ai/analytics/dashboard/
    Get dashboard summary.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_dashboard_data()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class EmotionTrendsView(BaseAIView):
    """
    GET /api/ai/analytics/emotions/
    Get emotion trends.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            period = request.query_params.get('period', 'week')
            result = services.get_emotion_trends(period=period)
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ProductivityTrendsView(BaseAIView):
    """
    GET /api/ai/analytics/productivity/
    Get productivity trends.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            period = request.query_params.get('period', 'week')
            result = services.get_productivity_trends(period=period)
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnalyticsReportView(BaseAIView):
    """
    GET /api/ai/analytics/report/
    Get comprehensive analytics report.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            period = request.query_params.get('period', 'month')
            result = services.get_analytics_report(period=period)
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CorrelationAnalysisView(BaseAIView):
    """
    GET /api/ai/analytics/correlation/
    Get emotion-productivity correlation.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            period = request.query_params.get('period', 'month')
            result = services.get_correlation_analysis(period=period)
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RiskAssessmentView(BaseAIView):
    """
    GET /api/ai/anomaly/risk/
    Get current risk assessment.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_risk_assessment()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnomalySummaryView(BaseAIView):
    """
    GET /api/ai/anomaly/summary/
    Get anomaly summary.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            days = int(request.query_params.get('days', 7))
            result = services.get_anomaly_summary(days=days)
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class WellnessInsightsView(BaseAIView):
    """
    GET /api/ai/wellness/insights/
    Get wellness insights.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_wellness_insights()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExplainRecommendationView(BaseAIView):
    """
    POST /api/ai/explain/
    Get explanation for AI recommendation.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            
            recommendation_type = request.data.get('type')
            recommendation = request.data.get('recommendation', {})
            context = request.data.get('context', {})
            
            if not recommendation_type:
                return Response(
                    APIResponseFormatter.error('recommendation type required'),
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = services.explain_recommendation(
                recommendation_type=recommendation_type,
                recommendation=recommendation,
                context=context
            )
            return Response(APIResponseFormatter.success(result))
            
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TrainModelsView(BaseAIView):
    """
    POST /api/ai/models/train/
    Train all AI models.
    """
    
    def post(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.train_models()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ServiceStatusView(BaseAIView):
    """
    GET /api/ai/status/
    Get status of all AI services.
    """
    
    def get(self, request):
        try:
            services = self.get_ai_services(request)
            result = services.get_service_status()
            return Response(APIResponseFormatter.success(result))
        except Exception as e:
            return Response(
                APIResponseFormatter.error(str(e)),
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
