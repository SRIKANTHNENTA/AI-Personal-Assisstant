"""
Conversation Learning Engine
Learns from user interactions and builds a knowledge base for future responses
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher


class ConversationLearner:
    """
    Learns from conversations and generates intelligent responses based on past interactions
    """
    
    def __init__(self, knowledge_base_path: str = None):
        if knowledge_base_path is None:
            knowledge_base_path = Path(__file__).parent / 'learned_knowledge.json'
        self.knowledge_base_path = Path(knowledge_base_path)
        self.static_knowledge_path = Path(__file__).parent / 'static_knowledge.json'
        self.knowledge = self._load_knowledge()
    
    def _load_knowledge(self) -> Dict:
        """Load existing learned and static knowledge"""
        base_data = {'qa_pairs': [], 'topics': {}, 'user_patterns': []}
        
        # Load learned knowledge
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    learned = json.load(f)
                    base_data['qa_pairs'].extend(learned.get('qa_pairs', []))
                    base_data['topics'].update(learned.get('topics', {}))
            except Exception as e:
                print(f"Error loading learned knowledge: {e}")
        
        # Load static knowledge (expanded technical/medical/etc.)
        if self.static_knowledge_path.exists():
            try:
                with open(self.static_knowledge_path, 'r', encoding='utf-8') as f:
                    static = json.load(f)
                    base_data['qa_pairs'].extend(static.get('qa_pairs', []))
                    # Merge topics if necessary, or just rely on QA pairs
            except Exception as e:
                print(f"Error loading static knowledge: {e}")
                
        return base_data
    
    def _save_knowledge(self):
        """Save ONLY learned knowledge to file (exclude static)"""
        try:
            # Filter out static knowledge before saving
            learned_qa = [p for p in self.knowledge['qa_pairs'] if p.get('source') != 'static']
            data_to_save = {
                'qa_pairs': learned_qa,
                'topics': self.knowledge['topics'],
                'user_patterns': self.knowledge.get('user_patterns', [])
            }
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving knowledge: {e}")
    
    def learn_from_interaction(self, user_question: str, ai_response: str, 
                                response_source: str = 'openai'):
        """
        Learn from a user-AI interaction
        Stores Q&A pairs for future reference
        """
        # Only learn from successful OpenAI responses or template responses
        if response_source not in ['openai', 'template', 'learning']:
            return
            
        # SAFETY CHECK: Don't learn if response contains common API error signatures
        error_signatures = [
            'Error code:', 'invalid_api_key', 'insufficient_quota', 
            'rate_limit_exceeded', 'module not found', 'traceback',
            'I\'m having trouble connecting right now'
        ]
        if any(sig.lower() in ai_response.lower() for sig in error_signatures):
            print(f"Skipping learning from error response: {ai_response[:50]}...")
            return
        
        # Normalize question
        normalized_question = self._normalize_text(user_question)
        
        # Check if similar question already exists
        existing_pair = self._find_similar_qa(normalized_question)
        
        if existing_pair:
            # Update confidence score
            existing_pair['confidence'] = min(1.0, existing_pair['confidence'] + 0.1)
            existing_pair['usage_count'] = existing_pair.get('usage_count', 1) + 1
        else:
            # Add new Q&A pair
            qa_pair = {
                'question': user_question,
                'normalized_question': normalized_question,
                'answer': ai_response,
                'source': response_source,
                'confidence': 0.8,
                'usage_count': 1,
                'keywords': self._extract_keywords(user_question)
            }
            self.knowledge['qa_pairs'].append(qa_pair)
        
        # Extract and store topic
        self._extract_topic(user_question, ai_response)
        
        # Save updated knowledge
        self._save_knowledge()
    
    def find_learned_response(self, user_question: str) -> Optional[str]:
        """
        Find a learned response for a similar question
        Returns the response if found, None otherwise
        """
        normalized_question = self._normalize_text(user_question)
        
        # Find most similar Q&A pair
        best_match = None
        best_similarity = 0.0
        
        for qa_pair in self.knowledge['qa_pairs']:
            similarity = self._calculate_similarity(
                normalized_question, 
                qa_pair['normalized_question']
            )
            
            # Also check keyword overlap
            keyword_score = self._keyword_similarity(
                user_question.lower(),
                qa_pair['keywords']
            )
            
            # Combined score
            combined_score = (similarity * 0.7) + (keyword_score * 0.3)
            
            if combined_score > best_similarity and combined_score > 0.75:
                best_similarity = combined_score
                best_match = qa_pair
        
        if best_match:
            # Update usage count
            best_match['usage_count'] = best_match.get('usage_count', 0) + 1
            self._save_knowledge()
            
            return best_match['answer']
        
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 
                     'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'i', 
                     'you', 'what', 'how', 'when', 'where', 'why'}
        
        words = self._normalize_text(text).split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:10]  # Top 10 keywords
    
    def _keyword_similarity(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword overlap score"""
        text_words = set(self._normalize_text(text).split())
        keyword_set = set(keywords)
        
        if not keyword_set:
            return 0.0
        
        overlap = len(text_words.intersection(keyword_set))
        return overlap / len(keyword_set)
    
    def _find_similar_qa(self, normalized_question: str) -> Optional[Dict]:
        """Find if a very similar question already exists"""
        for qa_pair in self.knowledge['qa_pairs']:
            if self._calculate_similarity(normalized_question, 
                                         qa_pair['normalized_question']) > 0.95:
                return qa_pair
        return None
    
    def _extract_topic(self, question: str, answer: str):
        """Extract and store topic from conversation"""
        keywords = self._extract_keywords(question)
        
        for keyword in keywords:
            if keyword not in self.knowledge['topics']:
                self.knowledge['topics'][keyword] = {
                    'count': 1,
                    'related_questions': [question[:100]]
                }
            else:
                self.knowledge['topics'][keyword]['count'] += 1
                if question[:100] not in self.knowledge['topics'][keyword]['related_questions']:
                    self.knowledge['topics'][keyword]['related_questions'].append(question[:100])
    
    def get_knowledge_stats(self) -> Dict:
        """Get statistics about learned knowledge"""
        return {
            'total_qa_pairs': len(self.knowledge['qa_pairs']),
            'total_topics': len(self.knowledge['topics']),
            'most_common_topics': sorted(
                self.knowledge['topics'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]
        }
