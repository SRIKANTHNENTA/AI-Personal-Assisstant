"""
Adaptive Personality Engine
Evolves AI assistant personality based on long-term interaction history.
Learns user preferences and adapts communication style dynamically.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path


class AdaptivePersonalityEngine:
    """
    Adaptive Personality Engine that evolves the AI assistant's communication
    style and behavior based on long-term user interactions.
    """
    
    # Personality dimensions (Big Five + custom)
    PERSONALITY_DIMENSIONS = {
        'formality': {'min': 0, 'max': 1, 'default': 0.5},  # Casual ↔ Formal
        'verbosity': {'min': 0, 'max': 1, 'default': 0.5},  # Concise ↔ Detailed
        'empathy': {'min': 0, 'max': 1, 'default': 0.7},    # Analytical ↔ Empathetic
        'proactivity': {'min': 0, 'max': 1, 'default': 0.6},  # Reactive ↔ Proactive
        'humor': {'min': 0, 'max': 1, 'default': 0.3},      # Serious ↔ Playful
        'encouragement': {'min': 0, 'max': 1, 'default': 0.7},  # Neutral ↔ Encouraging
        'directness': {'min': 0, 'max': 1, 'default': 0.6},  # Indirect ↔ Direct
    }
    
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        
        # Current personality state
        self.personality_state = {
            dim: props['default'] 
            for dim, props in self.PERSONALITY_DIMENSIONS.items()
        }
        
        # Learning rate for personality adaptation
        self.learning_rate = 0.05
        
        # Interaction history
        self.interaction_history = []
        self.feedback_history = []
        
        # User preference model
        self.user_preferences = {
            'preferred_greeting_style': None,
            'preferred_response_length': 'medium',  # short, medium, long
            'topics_of_interest': [],
            'communication_times': defaultdict(int),
            'emoji_preference': 0.5,  # 0 = no emojis, 1 = lots of emojis
        }
        
        # Model paths
        self.model_dir = Path(__file__).parent / 'personality_models'
        self.model_dir.mkdir(exist_ok=True)
        
        # Load existing personality
        self._load_personality()
        
        # Communication style templates
        self.style_templates = self._initialize_style_templates()
    
    def _load_personality(self):
        """Load saved personality state"""
        if self.user_id:
            path = self.model_dir / f'personality_{self.user_id}.json'
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        self.personality_state = data.get('personality_state', self.personality_state)
                        self.user_preferences = data.get('user_preferences', self.user_preferences)
                        self.interaction_history = data.get('interaction_history', [])[-500:]
                        self.feedback_history = data.get('feedback_history', [])[-100:]
                except Exception as e:
                    print(f"Warning: Could not load personality: {e}")
    
    def _save_personality(self):
        """Save personality state"""
        if self.user_id:
            path = self.model_dir / f'personality_{self.user_id}.json'
            try:
                data = {
                    'personality_state': self.personality_state,
                    'user_preferences': dict(self.user_preferences),
                    'interaction_history': self.interaction_history[-500:],
                    'feedback_history': self.feedback_history[-100:],
                    'last_updated': datetime.now().isoformat()
                }
                # Convert defaultdict to regular dict
                if 'communication_times' in data['user_preferences']:
                    data['user_preferences']['communication_times'] = dict(
                        data['user_preferences']['communication_times']
                    )
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save personality: {e}")
    
    def _initialize_style_templates(self) -> Dict:
        """Initialize communication style templates"""
        return {
            'greetings': {
                'formal': ["Good {time_of_day}.", "Hello.", "Greetings."],
                'casual': ["Hey!", "Hi there!", "Hello!"],
                'enthusiastic': ["Great to see you!", "Hey, welcome back!", "Hello! 🎉"]
            },
            'acknowledgments': {
                'formal': ["Understood.", "Certainly.", "I will attend to that."],
                'casual': ["Got it!", "Sure thing!", "On it!"],
                'empathetic': ["I understand.", "I hear you.", "That makes sense."]
            },
            'encouragements': {
                'subtle': ["Keep going.", "You're on track.", "Good progress."],
                'enthusiastic': ["Great job!", "You're doing amazing!", "Fantastic work!"],
                'supportive': ["I believe in you.", "You've got this.", "You're capable."]
            },
            'task_completion': {
                'minimal': ["Done.", "Task completed.", "Finished."],
                'detailed': ["I've completed the task. Here's what was done:", "Task completed successfully. Details:"],
                'celebratory': ["All done! Great work getting this completed!", "Task finished! 🎯"]
            },
            'error_handling': {
                'formal': ["I apologize for the inconvenience.", "An error has occurred."],
                'casual': ["Oops! Something went wrong.", "Hmm, that didn't work."],
                'supportive': ["No worries, let's try another approach.", "Don't worry, we'll figure this out."]
            }
        }
    
    def record_interaction(self, 
                          user_message: str,
                          assistant_response: str,
                          emotion_state: str = None,
                          response_time: float = None) -> Dict:
        """
        Record an interaction for personality learning.
        
        Args:
            user_message: What the user said
            assistant_response: What the assistant replied
            emotion_state: User's detected emotional state
            response_time: How long the response took
            
        Returns:
            Dict with interaction analysis
        """
        now = datetime.now()
        
        # Analyze interaction
        analysis = self._analyze_interaction(user_message, assistant_response)
        
        # Record interaction
        interaction = {
            'timestamp': now.isoformat(),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'user_message_length': len(user_message),
            'response_length': len(assistant_response),
            'user_formality': analysis['user_formality'],
            'user_emoji_usage': analysis['user_emoji_usage'],
            'topic': analysis.get('topic'),
            'emotion_state': emotion_state,
            'response_time': response_time
        }
        
        self.interaction_history.append(interaction)
        
        # Update communication time preference
        time_key = f"{now.weekday()}_{now.hour}"
        if isinstance(self.user_preferences['communication_times'], defaultdict):
            self.user_preferences['communication_times'][time_key] += 1
        else:
            self.user_preferences['communication_times'] = defaultdict(int, self.user_preferences['communication_times'])
            self.user_preferences['communication_times'][time_key] += 1
        
        # Adapt personality based on interaction
        adaptation = self._adapt_to_interaction(analysis)
        
        # Save periodically
        if len(self.interaction_history) % 10 == 0:
            self._save_personality()
        
        return {
            'recorded': True,
            'analysis': analysis,
            'adaptation': adaptation
        }
    
    def _analyze_interaction(self, user_message: str, assistant_response: str) -> Dict:
        """Analyze interaction patterns"""
        # Check formality indicators
        formal_indicators = ['please', 'kindly', 'would you', 'could you', 'i would like']
        casual_indicators = ['hey', 'yo', 'wanna', 'gonna', 'lol', 'haha', '!']
        
        message_lower = user_message.lower()
        
        formal_count = sum(1 for ind in formal_indicators if ind in message_lower)
        casual_count = sum(1 for ind in casual_indicators if ind in message_lower)
        
        if formal_count > casual_count:
            user_formality = 0.7
        elif casual_count > formal_count:
            user_formality = 0.3
        else:
            user_formality = 0.5
        
        # Check emoji usage
        emoji_count = sum(1 for c in user_message if ord(c) > 127000)
        user_emoji_usage = min(1.0, emoji_count / 3)
        
        # Estimate preferred response length
        if len(user_message) < 20:
            preferred_length = 'short'
        elif len(user_message) < 100:
            preferred_length = 'medium'
        else:
            preferred_length = 'long'
        
        # Detect topic (simple keyword-based)
        topic = 'general'
        if any(w in message_lower for w in ['task', 'todo', 'remind', 'schedule']):
            topic = 'task_management'
        elif any(w in message_lower for w in ['feel', 'mood', 'emotion', 'stress']):
            topic = 'emotional'
        elif any(w in message_lower for w in ['help', 'how', 'what', 'why']):
            topic = 'question'
        elif any(w in message_lower for w in ['project', 'ai', 'research', 'analysis', 'architecture']):
            topic = 'technical_project'
        
        return {
            'user_formality': user_formality,
            'user_emoji_usage': user_emoji_usage,
            'preferred_length': preferred_length,
            'topic': topic,
            'message_length': len(user_message)
        }
    
    def _adapt_to_interaction(self, analysis: Dict) -> Dict:
        """Adapt personality based on interaction analysis"""
        adaptations = {}
        
        # Adapt formality
        target_formality = analysis['user_formality']
        current_formality = self.personality_state['formality']
        new_formality = current_formality + self.learning_rate * (target_formality - current_formality)
        self.personality_state['formality'] = np.clip(new_formality, 0, 1)
        adaptations['formality'] = self.personality_state['formality']
        
        # Adapt emoji usage to humor dimension
        target_humor = analysis['user_emoji_usage']
        current_humor = self.personality_state['humor']
        new_humor = current_humor + (self.learning_rate / 2) * (target_humor - current_humor)
        self.personality_state['humor'] = np.clip(new_humor, 0, 1)
        adaptations['humor'] = self.personality_state['humor']
        
        # Adapt verbosity based on message length preference
        length_map = {'short': 0.3, 'medium': 0.5, 'long': 0.7}
        target_verbosity = length_map.get(analysis['preferred_length'], 0.5)
        current_verbosity = self.personality_state['verbosity']
        new_verbosity = current_verbosity + (self.learning_rate / 2) * (target_verbosity - current_verbosity)
        self.personality_state['verbosity'] = np.clip(new_verbosity, 0, 1)
        adaptations['verbosity'] = self.personality_state['verbosity']
        
        # Custom adaptation for technical projects
        if analysis.get('topic') == 'technical_project':
            # Boost formality, verbosity, and directness; decrease empathy (move towards analytical)
            self.personality_state['formality'] = np.clip(self.personality_state['formality'] + 0.1, 0, 1)
            self.personality_state['verbosity'] = np.clip(self.personality_state['verbosity'] + 0.2, 0, 1)
            self.personality_state['directness'] = np.clip(self.personality_state['directness'] + 0.1, 0, 1)
            self.personality_state['empathy'] = np.clip(self.personality_state['empathy'] - 0.1, 0, 1)
            adaptations['technical_boost'] = True
            
        return adaptations
    
    def process_feedback(self, feedback_type: str, feedback_value: float,
                        context: Dict = None) -> Dict:
        """
        Process explicit user feedback to adjust personality.
        
        Args:
            feedback_type: Type of feedback ('response_quality', 'tone', 'helpfulness', etc.)
            feedback_value: Value between -1 (negative) and 1 (positive)
            context: Additional context about what triggered the feedback
            
        Returns:
            Dict with adaptation results
        """
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'type': feedback_type,
            'value': feedback_value,
            'context': context
        }
        self.feedback_history.append(feedback)
        
        # Adapt based on feedback
        adaptations = {}
        
        if feedback_type == 'too_formal':
            self.personality_state['formality'] -= 0.1
            adaptations['formality'] = self.personality_state['formality']
        elif feedback_type == 'too_casual':
            self.personality_state['formality'] += 0.1
            adaptations['formality'] = self.personality_state['formality']
        elif feedback_type == 'too_long':
            self.personality_state['verbosity'] -= 0.15
            adaptations['verbosity'] = self.personality_state['verbosity']
        elif feedback_type == 'too_short':
            self.personality_state['verbosity'] += 0.15
            adaptations['verbosity'] = self.personality_state['verbosity']
        elif feedback_type == 'more_empathy':
            self.personality_state['empathy'] += 0.1
            adaptations['empathy'] = self.personality_state['empathy']
        elif feedback_type == 'less_empathy':
            self.personality_state['empathy'] -= 0.1
            adaptations['empathy'] = self.personality_state['empathy']
        elif feedback_type == 'response_quality':
            # Positive feedback reinforces current personality
            if feedback_value > 0:
                for dim in self.personality_state:
                    # Move slightly towards current position (reinforce)
                    pass
            else:
                # Negative feedback - try different approach
                self._explore_personality()
        
        # Clip all values
        for dim in self.personality_state:
            self.personality_state[dim] = np.clip(self.personality_state[dim], 0, 1)
        
        self._save_personality()
        
        return {
            'feedback_recorded': True,
            'adaptations': adaptations,
            'current_personality': self.personality_state.copy()
        }
    
    def _explore_personality(self):
        """Slightly randomize personality to explore better options"""
        for dim in self.personality_state:
            noise = np.random.uniform(-0.05, 0.05)
            self.personality_state[dim] = np.clip(
                self.personality_state[dim] + noise, 0, 1
            )
    
    def get_response_style(self, context: Dict = None) -> Dict:
        """
        Get current response style guidelines based on personality state.
        
        Args:
            context: Current context (emotion, topic, time, etc.)
            
        Returns:
            Style guidelines for response generation
        """
        context = context or {}
        
        # Determine greeting style
        if self.personality_state['formality'] > 0.6:
            greeting_style = 'formal'
        elif self.personality_state['formality'] < 0.4:
            greeting_style = 'casual'
        else:
            greeting_style = 'neutral'
        
        # Determine response length
        if self.personality_state['verbosity'] > 0.7:
            response_length = 'detailed'
        elif self.personality_state['verbosity'] < 0.3:
            response_length = 'concise'
        else:
            response_length = 'moderate'
        
        # Determine empathy level
        user_emotion = context.get('user_emotion', 'neutral')
        empathy_boost = 0
        if user_emotion in ['sad', 'stressed', 'anxious', 'angry']:
            empathy_boost = 0.2
        
        effective_empathy = min(1.0, self.personality_state['empathy'] + empathy_boost)
        
        # Determine if proactive suggestions should be made
        proactive = self.personality_state['proactivity'] > 0.5
        
        # Determine humor/emoji usage
        use_humor = self.personality_state['humor'] > 0.5
        use_emojis = self.personality_state['humor'] > 0.4 and context.get('user_used_emoji', False)
        
        # Get appropriate templates
        templates = {
            'greeting': self._select_template('greetings', greeting_style),
            'acknowledgment': self._select_template('acknowledgments', 
                'empathetic' if effective_empathy > 0.6 else greeting_style),
            'encouragement': self._select_template('encouragements',
                'enthusiastic' if self.personality_state['encouragement'] > 0.7 else 'subtle'),
            'task_completion': self._select_template('task_completion',
                'celebratory' if use_humor else ('detailed' if response_length == 'detailed' else 'minimal')),
            'error': self._select_template('error_handling',
                'supportive' if effective_empathy > 0.6 else greeting_style)
        }
        
        return {
            'greeting_style': greeting_style,
            'response_length': response_length,
            'empathy_level': effective_empathy,
            'be_proactive': proactive,
            'use_humor': use_humor,
            'use_emojis': use_emojis,
            'directness': self.personality_state['directness'],
            'templates': templates,
            'personality_snapshot': self.personality_state.copy()
        }
    
    def _select_template(self, category: str, style: str) -> str:
        """Select an appropriate template"""
        templates = self.style_templates.get(category, {})
        style_templates = templates.get(style, templates.get('casual', [""]))
        if style_templates:
            return np.random.choice(style_templates)
        return ""
    
    def generate_system_prompt(self, context: Dict = None) -> str:
        """
        Generate a system prompt that reflects current personality.
        
        Args:
            context: Current context for the conversation
            
        Returns:
            System prompt string for LLM
        """
        style = self.get_response_style(context)
        
        prompt_parts = [
            "You are a helpful AI personal assistant."
        ]
        
        # Identity Guard
        user_name = context.get('user_name', 'the user')
        prompt_parts.append(f"Identity Note: You are the Assistant. The person you are interacting with is {user_name}. Do not confuse yourself with the user.")
        
        # Add formality instruction
        if style['greeting_style'] == 'formal':
            prompt_parts.append("Communicate in a professional, formal manner.")
        elif style['greeting_style'] == 'casual':
            prompt_parts.append("Be friendly and casual in your communication.")
        
        # Add verbosity instruction
        if style['response_length'] == 'concise':
            prompt_parts.append("Keep responses brief and to the point.")
        elif style['response_length'] == 'detailed':
            prompt_parts.append("Provide thorough, detailed responses.")
        
        # Add empathy/analytical instruction
        if style['empathy_level'] > 0.7:
            prompt_parts.append("Show empathy and emotional understanding. Acknowledge feelings.")
        elif style['empathy_level'] < 0.4:
            prompt_parts.append("Provide highly analytical, expert-level insights. Focus on technical reasoning and structural logic.")
        
        # Add proactivity instruction
        if style['be_proactive']:
            prompt_parts.append("Proactively suggest helpful actions or insights when appropriate.")
        
        # Add humor instruction
        if style['use_humor']:
            prompt_parts.append("Feel free to use light humor when appropriate.")
        
        # Add directness instruction
        if style['directness'] > 0.7:
            prompt_parts.append("Be direct and straightforward in responses.")
        elif style['directness'] < 0.3:
            prompt_parts.append("Be diplomatic and gentle in phrasing.")
        
        return " ".join(prompt_parts)
    
    def get_personality_summary(self) -> Dict:
        """Get a summary of the current personality state"""
        # Describe each dimension
        descriptions = {}
        
        for dim, value in self.personality_state.items():
            if dim == 'formality':
                if value > 0.7:
                    descriptions[dim] = 'Professional and formal'
                elif value < 0.3:
                    descriptions[dim] = 'Casual and friendly'
                else:
                    descriptions[dim] = 'Balanced formality'
            
            elif dim == 'verbosity':
                if value > 0.7:
                    descriptions[dim] = 'Detailed and thorough'
                elif value < 0.3:
                    descriptions[dim] = 'Concise and brief'
                else:
                    descriptions[dim] = 'Moderate detail'
            
            elif dim == 'empathy':
                if value > 0.7:
                    descriptions[dim] = 'Highly empathetic'
                elif value < 0.3:
                    descriptions[dim] = 'Analytically focused'
                else:
                    descriptions[dim] = 'Balanced empathy'
            
            elif dim == 'proactivity':
                if value > 0.7:
                    descriptions[dim] = 'Proactively helpful'
                elif value < 0.3:
                    descriptions[dim] = 'Responsive to requests'
                else:
                    descriptions[dim] = 'Moderately proactive'
            
            elif dim == 'humor':
                if value > 0.7:
                    descriptions[dim] = 'Playful and fun'
                elif value < 0.3:
                    descriptions[dim] = 'Serious and focused'
                else:
                    descriptions[dim] = 'Occasionally playful'
            
            elif dim == 'encouragement':
                if value > 0.7:
                    descriptions[dim] = 'Highly encouraging'
                elif value < 0.3:
                    descriptions[dim] = 'Neutral feedback style'
                else:
                    descriptions[dim] = 'Supportive'
            
            elif dim == 'directness':
                if value > 0.7:
                    descriptions[dim] = 'Direct and straightforward'
                elif value < 0.3:
                    descriptions[dim] = 'Diplomatic and gentle'
                else:
                    descriptions[dim] = 'Balanced directness'
        
        # Overall personality type
        overall_type = self._determine_personality_type()
        
        return {
            'personality_state': self.personality_state.copy(),
            'descriptions': descriptions,
            'overall_type': overall_type,
            'total_interactions': len(self.interaction_history),
            'learning_progress': min(1.0, len(self.interaction_history) / 100)
        }
    
    def _determine_personality_type(self) -> str:
        """Determine overall personality type label"""
        state = self.personality_state
        
        # High empathy + high encouragement = Supportive
        if state['empathy'] > 0.6 and state['encouragement'] > 0.6:
            if state['humor'] > 0.5:
                return "Supportive & Fun"
            return "Supportive & Caring"
        
        # High formality + high directness = Professional
        if state['formality'] > 0.6 and state['directness'] > 0.6:
            return "Professional & Direct"
        
        # Low formality + high humor = Playful
        if state['formality'] < 0.4 and state['humor'] > 0.6:
            return "Friendly & Playful"
        
        # High verbosity + high proactivity = Comprehensive
        if state['verbosity'] > 0.6 and state['proactivity'] > 0.6:
            return "Thorough & Proactive"
        
        # Low verbosity + high directness = Efficient
        if state['verbosity'] < 0.4 and state['directness'] > 0.6:
            return "Efficient & Direct"
        
        return "Balanced & Adaptable"


class PersonalityEvolutionTracker:
    """
    Track and visualize personality evolution over time.
    """
    
    def __init__(self, personality_engine: AdaptivePersonalityEngine):
        self.engine = personality_engine
        self.snapshots = []
        self.max_snapshots = 1000
    
    def record_snapshot(self):
        """Record current personality snapshot"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'state': self.engine.personality_state.copy(),
            'interactions_count': len(self.engine.interaction_history)
        }
        self.snapshots.append(snapshot)
        
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def get_evolution_data(self, dimension: str = None) -> Dict:
        """
        Get personality evolution data for visualization.
        
        Args:
            dimension: Specific dimension to track, or None for all
            
        Returns:
            Time series data for visualization
        """
        if not self.snapshots:
            return {'success': False, 'error': 'No snapshots recorded'}
        
        if dimension:
            values = [
                {
                    'timestamp': s['timestamp'],
                    'value': s['state'].get(dimension, 0.5)
                }
                for s in self.snapshots
            ]
            return {
                'success': True,
                'dimension': dimension,
                'data': values
            }
        
        # All dimensions
        dimensions = list(self.engine.PERSONALITY_DIMENSIONS.keys())
        series = {dim: [] for dim in dimensions}
        
        for snapshot in self.snapshots:
            for dim in dimensions:
                series[dim].append({
                    'timestamp': snapshot['timestamp'],
                    'value': snapshot['state'].get(dim, 0.5)
                })
        
        return {
            'success': True,
            'dimensions': dimensions,
            'series': series
        }
    
    def get_stability_analysis(self) -> Dict:
        """Analyze how stable/volatile personality has been"""
        if len(self.snapshots) < 10:
            return {
                'success': False,
                'error': 'Insufficient data for stability analysis'
            }
        
        dimensions = list(self.engine.PERSONALITY_DIMENSIONS.keys())
        stability = {}
        
        for dim in dimensions:
            values = [s['state'].get(dim, 0.5) for s in self.snapshots]
            std_dev = np.std(values)
            
            if std_dev < 0.05:
                stability[dim] = {'stability': 'very_stable', 'std': std_dev}
            elif std_dev < 0.1:
                stability[dim] = {'stability': 'stable', 'std': std_dev}
            elif std_dev < 0.2:
                stability[dim] = {'stability': 'moderate', 'std': std_dev}
            else:
                stability[dim] = {'stability': 'volatile', 'std': std_dev}
        
        # Overall stability score
        avg_std = np.mean([s['std'] for s in stability.values()])
        overall_stability = 1.0 - min(1.0, avg_std * 5)
        
        return {
            'success': True,
            'dimension_stability': stability,
            'overall_stability_score': overall_stability,
            'snapshots_analyzed': len(self.snapshots)
        }
