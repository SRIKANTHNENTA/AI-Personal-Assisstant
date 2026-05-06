"""
Reinforcement Learning Task Scheduler
Optimizes task scheduling using Q-learning and policy gradient methods.
Learns optimal scheduling policies based on user productivity patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import random


class RLTaskScheduler:
    """
    Reinforcement Learning-based Task Scheduler.
    Uses Q-learning to optimize task scheduling decisions based on:
    - User's emotional state
    - Time of day
    - Task complexity
    - Historical performance
    """
    
    def __init__(self, user_id: str = None):
        self.user_id = user_id
        
        # State space parameters
        self.n_time_slots = 24  # Hours in a day
        self.n_emotion_states = 5  # Happy, Neutral, Sad, Stressed, Tired
        self.n_energy_levels = 3  # High, Medium, Low
        self.n_task_priorities = 3  # High, Medium, Low
        self.n_task_complexities = 3  # Complex, Medium, Simple
        
        # Action space: which time slot to schedule task
        self.n_actions = self.n_time_slots
        
        # Q-table dimensions
        state_dims = (
            self.n_time_slots,
            self.n_emotion_states,
            self.n_energy_levels,
            self.n_task_priorities,
            self.n_task_complexities
        )
        self.state_dims = state_dims
        
        # Initialize Q-table
        self.q_table = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Reward history
        self.reward_history = []
        self.episode_rewards = []
        
        # Model paths
        self.model_dir = Path(__file__).parent / 'rl_scheduler_models'
        self.model_dir.mkdir(exist_ok=True)
        
        # Load existing Q-table if available
        self._load_q_table()
        
        # Mappings
        self.emotion_to_idx = {
            'happy': 0, 'excited': 0, 'calm': 0,
            'neutral': 1,
            'sad': 2, 'anxious': 2,
            'stressed': 3, 'angry': 3,
            'tired': 4
        }
        
        self.energy_to_idx = {
            'high': 0,
            'medium': 1, 'moderate': 1,
            'low': 2
        }
        
        self.priority_to_idx = {
            'high': 0, 'critical': 0, 'urgent': 0,
            'medium': 1, 'normal': 1,
            'low': 2
        }
        
        self.complexity_to_idx = {
            'complex': 0, 'hard': 0, 'difficult': 0,
            'medium': 1, 'moderate': 1,
            'simple': 2, 'easy': 2
        }
    
    def _load_q_table(self):
        """Load Q-table from file"""
        if self.user_id:
            path = self.model_dir / f'q_table_{self.user_id}.json'
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # Convert string keys back to tuples
                        self.q_table = {
                            tuple(map(int, k.split(','))): v 
                            for k, v in data.get('q_table', {}).items()
                        }
                        self.epsilon = data.get('epsilon', 0.2)
                        self.reward_history = data.get('reward_history', [])
                except Exception as e:
                    print(f"Warning: Could not load Q-table: {e}")
    
    def _save_q_table(self):
        """Save Q-table to file"""
        if self.user_id:
            path = self.model_dir / f'q_table_{self.user_id}.json'
            try:
                data = {
                    'q_table': {
                        ','.join(map(str, k)): v 
                        for k, v in self.q_table.items()
                    },
                    'epsilon': self.epsilon,
                    'reward_history': self.reward_history[-1000:],
                    'last_updated': datetime.now().isoformat()
                }
                with open(path, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Warning: Could not save Q-table: {e}")
    
    def _encode_state(self, current_hour: int, emotion: str, 
                      energy: str, priority: str, complexity: str) -> tuple:
        """Encode state as tuple for Q-table lookup"""
        return (
            current_hour % self.n_time_slots,
            self.emotion_to_idx.get(emotion.lower(), 1),
            self.energy_to_idx.get(energy.lower(), 1),
            self.priority_to_idx.get(priority.lower(), 1),
            self.complexity_to_idx.get(complexity.lower(), 1)
        )
    
    def _get_q_values(self, state: tuple) -> np.ndarray:
        """Get Q-values for a state, initializing if necessary"""
        if state not in self.q_table:
            # Initialize with small random values
            self.q_table[state] = np.random.uniform(-0.1, 0.1, self.n_actions).tolist()
        return np.array(self.q_table[state])
    
    def get_optimal_time_slot(self, 
                              task: Dict,
                              user_state: Dict,
                              use_exploration: bool = True) -> Dict:
        """
        Get optimal time slot for a task using learned policy.
        
        Args:
            task: {'priority': str, 'complexity': str, 'estimated_duration': int}
            user_state: {'emotion': str, 'energy': str, 'current_hour': int}
            use_exploration: Whether to use epsilon-greedy exploration
            
        Returns:
            Dict with recommended time slot and explanation
        """
        current_hour = user_state.get('current_hour', datetime.now().hour)
        emotion = user_state.get('emotion', 'neutral')
        energy = user_state.get('energy', 'medium')
        priority = task.get('priority', 'medium')
        complexity = task.get('complexity', 'medium')
        
        # Encode current state
        state = self._encode_state(current_hour, emotion, energy, priority, complexity)
        
        # Get Q-values
        q_values = self._get_q_values(state)
        
        # Apply time constraints (can't schedule in the past)
        for hour in range(current_hour):
            q_values[hour] = -float('inf')
        
        # Epsilon-greedy action selection
        if use_exploration and random.random() < self.epsilon:
            # Explore: random future time slot
            valid_hours = list(range(current_hour, 24))
            if not valid_hours:
                valid_hours = list(range(24))
            action = random.choice(valid_hours)
        else:
            # Exploit: best action
            action = int(np.argmax(q_values))
        
        # Generate explanation
        explanation = self._explain_recommendation(
            action, state, q_values, task, user_state
        )
        
        return {
            'success': True,
            'recommended_hour': action,
            'time_range': f"{action:02d}:00 - {(action + 1) % 24:02d}:00",
            'confidence': self._calculate_confidence(q_values, action),
            'q_value': float(q_values[action]) if q_values[action] != -float('inf') else 0,
            'alternative_slots': self._get_alternatives(q_values, current_hour, top_k=3),
            'explanation': explanation,
            'exploration_used': use_exploration and random.random() < self.epsilon
        }
    
    def _explain_recommendation(self, action: int, state: tuple, 
                                q_values: np.ndarray, task: Dict, 
                                user_state: Dict) -> str:
        """Generate human-readable explanation for recommendation"""
        hour = action
        
        # Time of day description
        if 5 <= hour < 9:
            time_desc = "early morning"
        elif 9 <= hour < 12:
            time_desc = "mid-morning"
        elif 12 <= hour < 14:
            time_desc = "midday"
        elif 14 <= hour < 17:
            time_desc = "afternoon"
        elif 17 <= hour < 21:
            time_desc = "evening"
        else:
            time_desc = "night"
        
        priority = task.get('priority', 'medium')
        complexity = task.get('complexity', 'medium')
        emotion = user_state.get('emotion', 'neutral')
        energy = user_state.get('energy', 'medium')
        
        # Build explanation
        reasons = []
        
        # Complexity-time matching
        if complexity.lower() in ['complex', 'hard'] and 9 <= hour <= 11:
            reasons.append("complex tasks perform best in mid-morning when focus is highest")
        elif complexity.lower() in ['simple', 'easy'] and (hour < 9 or hour > 17):
            reasons.append("simple tasks can be done outside peak hours")
        
        # Energy-time matching
        if energy.lower() == 'low' and hour in [13, 14, 15]:
            reasons.append("post-lunch dip may affect energy")
        elif energy.lower() == 'high' and 9 <= hour <= 11:
            reasons.append("high energy aligns with peak productivity hours")
        
        # Priority-urgency matching
        if priority.lower() == 'high':
            reasons.append("high-priority task scheduled at earliest suitable slot")
        
        if not reasons:
            reasons.append(f"based on learned patterns for {time_desc} productivity")
        
        return f"Recommended {time_desc} ({hour:02d}:00) because {'; '.join(reasons)}."
    
    def _calculate_confidence(self, q_values: np.ndarray, action: int) -> float:
        """Calculate confidence in recommendation"""
        valid_q = q_values[q_values != -float('inf')]
        if len(valid_q) == 0:
            return 0.5
        
        # Softmax-like confidence
        current_q = q_values[action]
        if current_q == -float('inf'):
            return 0.0
        
        max_q = max(valid_q)
        min_q = min(valid_q)
        
        if max_q == min_q:
            return 0.5
        
        return (current_q - min_q) / (max_q - min_q)
    
    def _get_alternatives(self, q_values: np.ndarray, 
                         current_hour: int, top_k: int = 3) -> List[Dict]:
        """Get alternative time slots"""
        alternatives = []
        
        for hour in range(self.n_actions):
            if hour >= current_hour and q_values[hour] != -float('inf'):
                alternatives.append({
                    'hour': hour,
                    'time_range': f"{hour:02d}:00 - {(hour + 1) % 24:02d}:00",
                    'q_value': float(q_values[hour])
                })
        
        # Sort by Q-value and return top-k
        alternatives.sort(key=lambda x: x['q_value'], reverse=True)
        return alternatives[:top_k]
    
    def record_outcome(self, 
                      task: Dict,
                      user_state: Dict,
                      scheduled_hour: int,
                      outcome: Dict) -> Dict:
        """
        Record task outcome for learning.
        
        Args:
            task: Original task details
            user_state: User state at scheduling time
            scheduled_hour: Hour the task was scheduled for
            outcome: {
                'completed': bool,
                'on_time': bool,
                'duration_accuracy': float,  # Predicted vs actual
                'user_satisfaction': float,  # 0-1
                'stress_level': float  # 0-1 during task
            }
            
        Returns:
            Dict with learning update info
        """
        # Calculate reward
        reward = self._calculate_reward(outcome)
        
        # Encode state
        current_hour = user_state.get('current_hour', scheduled_hour)
        state = self._encode_state(
            current_hour,
            user_state.get('emotion', 'neutral'),
            user_state.get('energy', 'medium'),
            task.get('priority', 'medium'),
            task.get('complexity', 'medium')
        )
        
        action = scheduled_hour
        
        # Q-learning update
        old_q = self._get_q_values(state)[action]
        
        # For terminal state (task completed), no future rewards
        new_q = old_q + self.learning_rate * (reward - old_q)
        
        # Update Q-table
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions
        self.q_table[state][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Record reward
        self.reward_history.append({
            'timestamp': datetime.now().isoformat(),
            'reward': reward,
            'state': list(state),
            'action': action
        })
        
        # Save periodically
        if len(self.reward_history) % 10 == 0:
            self._save_q_table()
        
        return {
            'success': True,
            'reward': reward,
            'old_q_value': old_q,
            'new_q_value': new_q,
            'epsilon': self.epsilon,
            'total_experiences': len(self.reward_history)
        }
    
    def _calculate_reward(self, outcome: Dict) -> float:
        """Calculate reward from task outcome"""
        reward = 0.0
        
        # Completion bonus
        if outcome.get('completed', False):
            reward += 5.0
        else:
            reward -= 3.0
        
        # On-time bonus
        if outcome.get('on_time', False):
            reward += 3.0
        else:
            reward -= 2.0
        
        # Duration accuracy (penalize poor estimates)
        duration_accuracy = outcome.get('duration_accuracy', 1.0)
        if 0.8 <= duration_accuracy <= 1.2:
            reward += 2.0
        else:
            reward -= abs(duration_accuracy - 1.0) * 2.0
        
        # User satisfaction
        satisfaction = outcome.get('user_satisfaction', 0.5)
        reward += (satisfaction - 0.5) * 4.0
        
        # Stress penalty
        stress = outcome.get('stress_level', 0.3)
        reward -= stress * 2.0
        
        return reward
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about learning progress"""
        if not self.reward_history:
            return {
                'status': 'no_data',
                'message': 'No learning experiences recorded yet'
            }
        
        rewards = [r['reward'] for r in self.reward_history]
        
        recent_rewards = rewards[-100:] if len(rewards) > 100 else rewards
        early_rewards = rewards[:100] if len(rewards) > 100 else rewards
        
        return {
            'total_experiences': len(self.reward_history),
            'unique_states': len(self.q_table),
            'current_epsilon': self.epsilon,
            'average_reward': float(np.mean(rewards)),
            'recent_avg_reward': float(np.mean(recent_rewards)),
            'early_avg_reward': float(np.mean(early_rewards)),
            'improvement': float(np.mean(recent_rewards) - np.mean(early_rewards))
            if len(rewards) > 100 else 0.0,
            'max_reward': float(max(rewards)),
            'min_reward': float(min(rewards))
        }


class PolicyGradientScheduler:
    """
    Policy Gradient (REINFORCE) based task scheduler.
    Uses neural network policy for continuous state spaces.
    """
    
    def __init__(self, state_dim: int = 10, n_actions: int = 24):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.policy_network = None
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        self._build_network()
    
    def _build_network(self):
        """Build policy network"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.policy_network = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.n_actions, activation='softmax')
            ])
            
            self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            print("✅ Policy Gradient network built")
            
        except ImportError:
            print("⚠️ TensorFlow not available for Policy Gradient")
            self.policy_network = None
    
    def encode_state(self, user_state: Dict, task: Dict) -> np.ndarray:
        """Encode state into feature vector"""
        features = []
        
        # Time features (cyclical encoding)
        hour = user_state.get('current_hour', 12)
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        day = user_state.get('day_of_week', 0)
        features.append(np.sin(2 * np.pi * day / 7))
        features.append(np.cos(2 * np.pi * day / 7))
        
        # Emotion valence (-1 to 1)
        emotion_valence = {
            'happy': 0.8, 'excited': 0.9, 'calm': 0.3,
            'neutral': 0.0, 'sad': -0.5, 'anxious': -0.3,
            'stressed': -0.6, 'angry': -0.7, 'tired': -0.2
        }
        features.append(emotion_valence.get(user_state.get('emotion', 'neutral'), 0.0))
        
        # Energy level (0 to 1)
        energy_level = {'high': 1.0, 'medium': 0.5, 'low': 0.2}
        features.append(energy_level.get(user_state.get('energy', 'medium'), 0.5))
        
        # Task priority (0 to 1)
        priority_level = {'high': 1.0, 'medium': 0.5, 'low': 0.2}
        features.append(priority_level.get(task.get('priority', 'medium'), 0.5))
        
        # Task complexity (0 to 1)
        complexity_level = {'complex': 1.0, 'medium': 0.5, 'simple': 0.2}
        features.append(complexity_level.get(task.get('complexity', 'medium'), 0.5))
        
        # Normalize duration (fallback to 30 mins)
        duration_val = task.get('estimated_duration')
        if duration_val is None:
            duration_val = 30
        duration = float(duration_val) / 120.0
        features.append(min(duration, 1.0))
        
        # Deadline urgency (hours until due / 24)
        deadline_hours = task.get('deadline_hours', 24) / 24.0
        features.append(min(deadline_hours, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Get action (time slot) from policy network.
        
        Returns:
            Tuple of (action, probability)
        """
        if self.policy_network is None:
            # Fallback to random action
            action = random.randint(0, self.n_actions - 1)
            return action, 1.0 / self.n_actions
        
        import tensorflow as tf
        
        state_tensor = tf.expand_dims(state, 0)
        probs = self.policy_network(state_tensor, training=False)[0].numpy()
        
        if training:
            # Sample from distribution
            action = np.random.choice(self.n_actions, p=probs)
        else:
            # Take best action
            action = np.argmax(probs)
        
        return int(action), float(probs[action])
    
    def store_transition(self, state: np.ndarray, action: int, reward: float):
        """Store transition for episode"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train_episode(self) -> Dict:
        """Train on collected episode data using REINFORCE"""
        if self.policy_network is None or not self.states:
            return {'success': False, 'error': 'No data or model'}
        
        import tensorflow as tf
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Policy gradient update
        with tf.GradientTape() as tape:
            loss = 0
            for state, action, G in zip(self.states, self.actions, returns):
                state_tensor = tf.expand_dims(state, 0)
                probs = self.policy_network(state_tensor, training=True)
                
                action_prob = probs[0, action]
                loss -= tf.math.log(action_prob + 1e-8) * G
            
            loss /= len(self.states)
        
        # Apply gradients
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.policy_network.trainable_variables)
        )
        
        episode_reward = sum(self.rewards)
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return {
            'success': True,
            'loss': float(loss.numpy()),
            'episode_reward': episode_reward,
            'episode_length': len(returns)
        }


class ContextAwareScheduler:
    """
    Context-aware task scheduler that combines multiple signals.
    Integrates RL scheduling with domain knowledge.
    """
    
    def __init__(self, rl_scheduler: Optional[RLTaskScheduler] = None, user_id: str = None):
        self.user_id = user_id
        self.rl_scheduler = rl_scheduler or RLTaskScheduler(user_id)
        
        # Domain knowledge: optimal times for different task types
        self.task_type_preferences = {
            'creative': [9, 10, 11, 20, 21],  # Morning and late evening
            'analytical': [9, 10, 11, 14, 15],  # Morning and mid-afternoon
            'routine': [8, 12, 17],  # Start of work, lunch, end of day
            'collaborative': [10, 11, 14, 15],  # Late morning and afternoon
            'deep_work': [9, 10, 11],  # Peak focus hours
            'admin': [12, 17, 18],  # Low energy times OK
        }
        
        # Energy curve (typical circadian rhythm)
        self.energy_curve = {
            0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
            6: 0.4, 7: 0.6, 8: 0.8, 9: 0.95, 10: 1.0, 11: 0.95,
            12: 0.7, 13: 0.5, 14: 0.6, 15: 0.8, 16: 0.85, 17: 0.8,
            18: 0.7, 19: 0.6, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.2
        }
    
    def schedule_task(self, 
                      task: Dict,
                      user_state: Dict,
                      constraints: Dict = None) -> Dict:
        """
        Schedule a task using hybrid RL + domain knowledge approach.
        
        Args:
            task: Task details including type, priority, complexity, duration
            user_state: User's current emotional and energy state
            constraints: {'earliest_hour': int, 'latest_hour': int, 'blocked_hours': list}
            
        Returns:
            Optimal scheduling recommendation
        """
        constraints = constraints or {}
        
        # Get RL recommendation
        rl_result = self.rl_scheduler.get_optimal_time_slot(task, user_state)
        rl_hour = rl_result['recommended_hour']
        rl_confidence = rl_result['confidence']
        
        # Get domain knowledge recommendation
        domain_result = self._domain_knowledge_recommendation(task, user_state)
        domain_hours = domain_result['preferred_hours']
        
        # Apply constraints
        earliest = constraints.get('earliest_hour', 0)
        latest = constraints.get('latest_hour', 23)
        blocked = set(constraints.get('blocked_hours', []))
        
        # Score each valid hour
        hour_scores = {}
        for hour in range(24):
            if hour < earliest or hour > latest or hour in blocked:
                continue
            
            score = 0.0
            
            # RL component (weighted by experience)
            rl_weight = min(len(self.rl_scheduler.reward_history) / 50, 0.6)
            if hour == rl_hour:
                score += rl_weight * rl_confidence
            
            # Domain knowledge component
            domain_weight = 1.0 - rl_weight
            if hour in domain_hours:
                score += domain_weight * 0.8
            
            # Energy curve match
            task_complexity = task.get('complexity', 'medium')
            if task_complexity in ['complex', 'hard']:
                score += 0.2 * self.energy_curve.get(hour, 0.5)
            
            # Urgency adjustment
            if task.get('priority', 'medium') == 'high':
                # Prefer earlier slots for high priority
                current_hour = user_state.get('current_hour', 12)
                if hour == current_hour or hour == current_hour + 1:
                    score += 0.2
            
            hour_scores[hour] = score
        
        if not hour_scores:
            return {
                'success': False,
                'error': 'No valid time slots available with given constraints'
            }
        
        # Get best hour
        best_hour = max(hour_scores, key=hour_scores.get)
        
        # Build detailed explanation
        explanation = self._build_explanation(
            best_hour, task, user_state, rl_result, domain_result
        )
        
        return {
            'success': True,
            'recommended_hour': best_hour,
            'time_range': f"{best_hour:02d}:00 - {(best_hour + 1) % 24:02d}:00",
            'score': hour_scores[best_hour],
            'rl_contribution': rl_result,
            'domain_contribution': domain_result,
            'all_scores': hour_scores,
            'explanation': explanation,
            'alternatives': sorted(
                [{'hour': h, 'score': s} for h, s in hour_scores.items()],
                key=lambda x: x['score'],
                reverse=True
            )[:5]
        }
    
    def _domain_knowledge_recommendation(self, task: Dict, user_state: Dict) -> Dict:
        """Get domain knowledge based recommendation"""
        task_type = task.get('type', 'general')
        
        # Get preferred hours for task type
        preferred_hours = self.task_type_preferences.get(
            task_type, 
            [9, 10, 11, 14, 15]  # Default to productive hours
        )
        
        # Adjust for user energy state
        energy = user_state.get('energy', 'medium')
        if energy == 'low':
            # Prefer later hours when energy might recover
            preferred_hours = [h for h in preferred_hours if h >= 10]
            if not preferred_hours:
                preferred_hours = [14, 15, 16]
        
        return {
            'preferred_hours': preferred_hours,
            'task_type': task_type,
            'energy_adjustment': energy
        }
    
    def _build_explanation(self, hour: int, task: Dict, user_state: Dict,
                          rl_result: Dict, domain_result: Dict) -> str:
        """Build comprehensive explanation"""
        parts = [f"Scheduling at {hour:02d}:00"]
        
        reasons = []
        
        # RL reasoning
        if hour == rl_result['recommended_hour']:
            reasons.append("aligns with learned optimal patterns")
        
        # Domain knowledge
        if hour in domain_result['preferred_hours']:
            reasons.append(f"optimal for {domain_result['task_type']} tasks")
        
        # Energy reasoning
        energy_level = self.energy_curve.get(hour, 0.5)
        complexity = task.get('complexity', 'medium')
        if complexity in ['complex', 'hard'] and energy_level > 0.8:
            reasons.append("peak energy for complex work")
        elif complexity in ['simple', 'easy'] and energy_level < 0.6:
            reasons.append("saves peak hours for harder tasks")
        
        if reasons:
            parts.append("because " + "; ".join(reasons))
        
        return ". ".join(parts) + "."
    
    def batch_schedule(self, tasks: List[Dict], user_state: Dict) -> List[Dict]:
        """
        Schedule multiple tasks optimally.
        
        Args:
            tasks: List of tasks to schedule
            user_state: Current user state
            
        Returns:
            List of scheduling recommendations
        """
        # Sort tasks by priority and deadline
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_tasks = sorted(
            enumerate(tasks),
            key=lambda x: (
                priority_order.get(x[1].get('priority', 'medium'), 1),
                x[1].get('deadline_hours') or 24
            )
        )
        
        scheduled = []
        blocked_hours = set()
        
        for orig_idx, task in sorted_tasks:
            # Schedule task avoiding already scheduled slots
            result = self.schedule_task(
                task,
                user_state,
                constraints={'blocked_hours': list(blocked_hours)}
            )
            
            if result['success']:
                hour = result['recommended_hour']
                duration = task.get('estimated_duration') or 30
                duration_hours = max(1, int(duration) // 60)
                
                # Block hours for this task
                for h in range(hour, min(hour + duration_hours, 24)):
                    blocked_hours.add(h)
                
                scheduled.append({
                    'task_index': orig_idx,
                    'task': task,
                    'schedule': result
                })
            else:
                scheduled.append({
                    'task_index': orig_idx,
                    'task': task,
                    'schedule': result
                })
        
        # Sort back to original order
        scheduled.sort(key=lambda x: x['task_index'])
        
        return scheduled
