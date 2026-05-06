"""
Hybrid response pipeline for chatbot replies.

Priority order:
1) Local knowledge base (JSON intents + optional neural intent classifier)
2) External LLM providers (Gemini/OpenAI)
3) Graceful fallback (never expose raw provider errors)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

try:
    from ai_services.gemini_handler import ask_gemini
    from ai_services.llm_handler import ask_llm
except ModuleNotFoundError:
    # Support running/importing this module outside package context.
    try:
        from .gemini_handler import ask_gemini
        from .llm_handler import ask_llm
    except ImportError:
        from gemini_handler import ask_gemini
        from llm_handler import ask_llm

logger = logging.getLogger(__name__)
_KB_WRITE_LOCK = Lock()


@dataclass
class MatchResult:
    """Represents a local knowledge-base match."""

    response: str
    source: str
    confidence: float
    intent: Optional[str] = None


class LocalKnowledgeBaseMatcher:
    """Matches user input against local intents via lexical and semantic search."""

    _SEMANTIC_STOPWORDS = {
        "what", "who", "where", "when", "why", "how", "which", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with", "about",
        "tell", "me", "please", "explain", "define", "i", "you", "my", "your",
    }

    def __init__(self, intents_path: Optional[Path] = None, semantic_threshold: float = 0.62):
        self.intents_path = intents_path or self._resolve_intents_path()
        self.semantic_threshold = semantic_threshold
        self.enable_neural_classifier = os.getenv("ENABLE_NEURAL_INTENT_CLASSIFIER", "false").lower() in {"1", "true", "yes"}
        self._intents_mtime: Optional[float] = None
        self.intents: Dict = {"intents": []}
        self._patterns: List[str] = []
        self._pattern_meta: List[Tuple[str, List[str]]] = []

        self._vectorizer = None
        self._pattern_vectors = None
        self._neural_chatbot = None

        self._load_intents()
        self._build_semantic_index()

    def _resolve_intents_path(self) -> Path:
        base_dir = Path(__file__).resolve().parent
        configured_path = os.getenv("CHATBOT_INTENTS_PATH", "").strip()
        if configured_path:
            candidate = Path(configured_path)
            if not candidate.is_absolute():
                candidate = (base_dir.parent / candidate).resolve()
            if candidate.exists():
                return candidate

        candidates = [
            base_dir / "intents_example.json",
            base_dir / "neural_model" / "intents.json",
            base_dir / "intent_taxonomy.json",  # fallback candidate if deployment differs
            base_dir.parent / "Chatbot" / "chatbot_codes" / "intents.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Default to canonical path even if missing; caller handles empty data safely.
        return base_dir / "neural_model" / "intents.json"

    def _load_intents(self) -> None:
        # Reset in-memory state so refresh operations do not duplicate entries.
        self.intents = {"intents": []}
        self._patterns = []
        self._pattern_meta = []

        if not self.intents_path.exists():
            logger.warning("Intents file not found at %s", self.intents_path)
            self._intents_mtime = None
            return

        try:
            with open(self.intents_path, "r", encoding="utf-8") as file:
                self.intents = json.load(file)
        except Exception as exc:
            logger.exception("Failed to load intents JSON: %s", exc)
            self.intents = {"intents": []}
            self._intents_mtime = None
            return

        try:
            self._intents_mtime = self.intents_path.stat().st_mtime
        except Exception:
            self._intents_mtime = None

        for intent in self.intents.get("intents", []):
            tag = intent.get("tag", "unknown")
            responses = intent.get("responses", [])
            for pattern in intent.get("patterns", []):
                if isinstance(pattern, str) and pattern.strip():
                    normalized = self._normalize_text(pattern)
                    self._patterns.append(normalized)
                    self._pattern_meta.append((tag, responses))

    def reload_intents(self) -> int:
        """Reload intents JSON from disk and rebuild semantic index if enabled."""
        self._load_intents()
        self._build_semantic_index()
        return len(self.intents.get("intents", []))

    def _ensure_intents_fresh(self) -> None:
        """Hot-reload intents when the JSON file changes on disk."""
        if not self.intents_path.exists():
            return
        try:
            current_mtime = self.intents_path.stat().st_mtime
        except Exception:
            return

        if self._intents_mtime is None or current_mtime > self._intents_mtime:
            self.reload_intents()

    def _build_semantic_index(self) -> None:
        if not self._patterns:
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self._pattern_vectors = self._vectorizer.fit_transform(self._patterns)
        except Exception as exc:
            # Semantic matching is optional; lexical matching still works.
            logger.info("Semantic index unavailable, using lexical intent matching only: %s", exc)
            self._vectorizer = None
            self._pattern_vectors = None

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _select_response(self, responses: List[str], user_input: str) -> Optional[str]:
        """Pick a context-aware response instead of purely random selection."""
        if not responses:
            return None

        query = self._normalize_text(user_input)
        definition_like = query.startswith(("what is", "who is", "explain", "define", "meaning of"))

        # For definitional questions, prioritize responses that read like definitions.
        if definition_like:
            if responses[0].strip():
                return responses[0]

            preferred_markers = [
                " is ",
                " means ",
                " refers ",
                "branch",
                "framework",
                "subset",
                "uses",
                "enables",
            ]
            scored = []
            for resp in responses:
                text = resp.lower()
                score = sum(1 for marker in preferred_markers if marker in text)
                # Slight preference for more informative lines.
                score += min(len(text) / 120.0, 1.0)
                scored.append((score, resp))
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1]

        return random.choice(responses)

    def _token_overlap_score(self, user_text: str, pattern_text: str) -> float:
        user_tokens = set(re.findall(r"[a-z0-9]+", user_text))
        pattern_tokens = set(re.findall(r"[a-z0-9]+", pattern_text))
        if not user_tokens or not pattern_tokens:
            return 0.0

        overlap = user_tokens.intersection(pattern_tokens)
        return len(overlap) / max(len(pattern_tokens), 1)

    def _content_tokens(self, text: str) -> set:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return {token for token in tokens if token not in self._SEMANTIC_STOPWORDS and len(token) > 2}

    def _response_for_tag(self, tag: str) -> Optional[str]:
        for intent in self.intents.get("intents", []):
            if intent.get("tag") == tag:
                responses = intent.get("responses", [])
                if responses:
                    return random.choice(responses)
        return None

    def _classifier_match(self, user_input: str) -> Optional[MatchResult]:
        # Lazy-load the neural model because it may be heavy on cold start.
        if self._neural_chatbot is None:
            try:
                try:
                    from ai_services.neural_chatbot import NeuralChatbot
                except ModuleNotFoundError:
                    try:
                        from .neural_chatbot import NeuralChatbot
                    except ImportError:
                        from neural_chatbot import NeuralChatbot

                self._neural_chatbot = NeuralChatbot()
            except Exception as exc:
                logger.info("Neural classifier unavailable: %s", exc)
                self._neural_chatbot = False  # sentinel to avoid repeated retries

        if not self._neural_chatbot:
            return None

        try:
            prediction = self._neural_chatbot.predict_intent(user_input, confidence_threshold=0.72)
            if not prediction:
                return None

            response = self._response_for_tag(prediction["intent"])
            if not response:
                response = self._neural_chatbot.get_response(user_input)
            if not response:
                return None

            return MatchResult(
                response=response,
                source="local_intent_classifier",
                confidence=float(prediction.get("confidence", 0.75)),
                intent=prediction.get("intent"),
            )
        except Exception as exc:
            logger.info("Classifier match failed: %s", exc)
            return None

    def _lexical_match(self, user_input: str) -> Optional[MatchResult]:
        if not self._patterns:
            return None

        query = self._normalize_text(user_input)

        # Score exact/containment candidates and choose the strongest instead of first hit.
        best_idx = -1
        best_score = 0.0
        for idx, pattern in enumerate(self._patterns):
            if query == pattern:
                score = 1.0
            elif pattern in query and len(pattern) >= 4:
                # Favor more specific pattern phrases.
                specificity = len(pattern.split()) / max(1, len(query.split()))
                overlap = self._token_overlap_score(query, pattern)
                score = 0.82 + 0.12 * min(1.0, specificity) + 0.06 * overlap
            else:
                continue

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            tag, responses = self._pattern_meta[best_idx]
            if responses:
                return MatchResult(
                    response=self._select_response(responses, user_input) or random.choice(responses),
                    source="local_keyword_match",
                    confidence=min(best_score, 0.99),
                    intent=tag,
                )

        # Token overlap for non-exact phrasings.
        best_index = -1
        best_score = 0.0
        for idx, pattern in enumerate(self._patterns):
            score = self._token_overlap_score(query, pattern)
            if score > best_score:
                best_score = score
                best_index = idx

        if best_index >= 0 and best_score >= 0.8:
            tag, responses = self._pattern_meta[best_index]
            if responses:
                return MatchResult(
                    response=self._select_response(responses, user_input) or random.choice(responses),
                    source="local_keyword_match",
                    confidence=min(0.9, best_score),
                    intent=tag,
                )

        return None

    def _semantic_match(self, user_input: str) -> Optional[MatchResult]:
        if self._vectorizer is None or self._pattern_vectors is None:
            return None

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            query = self._normalize_text(user_input)
            query_vector = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self._pattern_vectors)[0]
            best_index = int(scores.argmax())
            best_score = float(scores[best_index])

            if best_score < self.semantic_threshold:
                return None

            best_pattern = self._patterns[best_index]
            query_content = self._content_tokens(query)
            pattern_content = self._content_tokens(best_pattern)
            overlap_count = len(query_content.intersection(pattern_content))
            overlap_ratio = overlap_count / max(1, len(pattern_content))

            # Prevent wrong-topic matches that share only generic tokens (e.g., "India").
            if overlap_count == 0:
                return None

            query_is_question = query.startswith(("what", "who", "where", "when", "why", "how", "which"))
            if query_is_question and overlap_count < 2 and best_score < 0.78:
                return None

            if overlap_ratio < 0.25 and best_score < 0.75:
                return None

            tag, responses = self._pattern_meta[best_index]
            if not responses:
                return None

            return MatchResult(
                response=self._select_response(responses, user_input) or random.choice(responses),
                source="local_semantic_match",
                confidence=best_score,
                intent=tag,
            )
        except Exception as exc:
            logger.info("Semantic match failed: %s", exc)
            return None

    def search(self, user_input: str) -> Optional[MatchResult]:
        self._ensure_intents_fresh()

        # Local-first policy: lexical -> semantic -> neural classifier.
        lexical = self._lexical_match(user_input)
        if lexical:
            return lexical

        semantic = self._semantic_match(user_input)
        if semantic:
            return semantic

        if self.enable_neural_classifier:
            classifier = self._classifier_match(user_input)
            if classifier:
                return classifier

        return None


class ExternalLLMClient:
    """LLM provider abstraction with provider failover and safe error handling."""

    def __init__(self, preferred_provider: Optional[str] = None):
        provider = (preferred_provider or os.getenv("PRIMARY_LLM_PROVIDER") or "gemini").lower().strip()
        if provider not in {"gemini", "openai"}:
            provider = "gemini"
        self.preferred_provider = provider

    def _has_provider_key(self, provider: str) -> bool:
        if provider == "gemini":
            key = os.getenv("GOOGLE_API_KEY", "")
            return bool(key and "AIza" in key)

        key = os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENAPI_API_KEY", "")
        return bool(key and "sk-" in key)

    def _provider_order(self) -> List[str]:
        secondary = "openai" if self.preferred_provider == "gemini" else "gemini"
        candidates = [self.preferred_provider, secondary]

        available = [p for p in candidates if self._has_provider_key(p)]
        if available:
            return available

        # If no key detected, keep original order so upstream fallback handling remains deterministic.
        return candidates

    def _looks_like_error_response(self, text: str) -> bool:
        if not text:
            return True

        normalized = text.lower()
        suspicious_markers = [
            "error:",
            "api key",
            "quota",
            "invalid",
            "having trouble connecting",
            "unable to",
            "⚠",
        ]
        return any(marker in normalized for marker in suspicious_markers)

    def generate(self, user_input: str, history: Optional[List[Dict]], system_prompt: Optional[str]) -> Tuple[str, str]:
        history = history or []

        for provider in self._provider_order():
            try:
                if provider == "gemini":
                    result = ask_gemini(user_input=user_input, history=history, system_prompt=system_prompt)
                else:
                    result = ask_llm(user_input=user_input, history=history, system_prompt=system_prompt)

                if result and not self._looks_like_error_response(result):
                    return result, provider

                logger.warning("Provider %s returned non-usable response", provider)
            except Exception as exc:
                logger.warning("Provider %s call failed: %s", provider, exc)

        raise RuntimeError("No LLM provider available")


class HybridResponsePipeline:
    """Production-ready response orchestration for rule-based + ML + LLM hybrid AI."""

    def __init__(self, kb_matcher: Optional[LocalKnowledgeBaseMatcher] = None, llm_client: Optional[ExternalLLMClient] = None):
        self.kb_matcher = kb_matcher or LocalKnowledgeBaseMatcher()
        self.llm_client = llm_client or ExternalLLMClient()
        self.enable_internet_enrichment = os.getenv("ENABLE_INTERNET_ENRICHMENT", "true").lower() in {"1", "true", "yes"}

    def search_local_knowledge_base(self, user_input: str) -> Optional[Dict[str, object]]:
        match = self.kb_matcher.search(user_input)
        if not match:
            return None

        return {
            "response": match.response,
            "source": match.source,
            "intent": match.intent,
            "confidence": match.confidence,
            "used_api": False,
        }

    def refresh_knowledge_base(self) -> Dict[str, object]:
        """Reload local intents from disk and return refresh metadata."""
        count = self.kb_matcher.reload_intents()
        return {
            "success": True,
            "intents_path": str(self.kb_matcher.intents_path),
            "intents_loaded": count,
        }

    def _slugify_tag(self, text: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        if not base:
            base = "query"
        return f"web_{base[:50]}"

    def _query_keywords(self, text: str) -> set:
        stopwords = {
            "what", "is", "are", "the", "a", "an", "can", "you", "please",
            "about", "tell", "me", "explain", "define", "of", "to", "in", "how",
            "who", "where", "when", "why", "and", "for",
        }
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in stopwords]
        return {t for t in tokens if len(t) >= 2}

    def _looks_relevant_internet_match(self, user_query: str, title: str, extract: str) -> bool:
        query_terms = self._query_keywords(user_query)
        if not query_terms:
            return False

        title_text = (title or "").lower()
        extract_text = (extract or "").lower()

        # Acronym-sensitive check: if query includes uppercase acronyms, require literal acronym hit.
        acronyms = re.findall(r"\b[A-Z]{2,}\b", user_query)
        for acronym in acronyms:
            if acronym.lower() not in title_text and acronym.lower() not in extract_text:
                return False

        hits = 0
        for term in query_terms:
            if term in title_text or re.search(rf"\b{re.escape(term)}\b", extract_text):
                hits += 1

        coverage = hits / max(1, len(query_terms))
        return hits >= 1 and coverage >= 0.5

    def _fetch_internet_knowledge(self, user_input: str) -> Optional[str]:
        """Fetch concise factual answer from the internet (Wikipedia)."""
        if not self.enable_internet_enrichment:
            return None

        query = user_input.strip()
        if len(query) < 3:
            return None

        try:
            import requests
        except Exception:
            logger.info("requests is unavailable; skipping internet enrichment")
            return None

        headers = {"User-Agent": "PersonalAssistantBot/1.0"}

        def build_query_candidates(text: str) -> List[str]:
            candidates: List[str] = []
            raw = text.strip()
            if raw:
                candidates.append(raw)

            normalized = self.kb_matcher._normalize_text(raw)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

            simplified = re.sub(
                r"^(what is|who is|what are|explain|define|tell me about|can you explain|please explain)\s+",
                "",
                normalized,
            ).strip()
            if simplified and simplified not in candidates:
                candidates.append(simplified)

            # Keep informative terms only for broad natural-language queries.
            stopwords = {
                "what", "is", "are", "the", "a", "an", "can", "you", "please",
                "about", "tell", "me", "explain", "define", "of", "to", "in",
            }
            tokens = [t for t in re.findall(r"[a-z0-9]+", normalized) if t not in stopwords]
            keyword_query = " ".join(tokens)
            if keyword_query and keyword_query not in candidates:
                candidates.append(keyword_query)

            return candidates

        try:
            title = None
            for candidate_query in build_query_candidates(query):
                search_resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": candidate_query,
                        "limit": 1,
                        "namespace": 0,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=8,
                )
                if search_resp.status_code != 200:
                    continue

                search_data = search_resp.json()
                titles = search_data[1] if isinstance(search_data, list) and len(search_data) > 1 else []
                if titles:
                    title = titles[0]
                    break

            if not title:
                return None

            summary_resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                headers=headers,
                timeout=8,
            )
            if summary_resp.status_code != 200:
                return None

            summary_data = summary_resp.json()
            if summary_data.get("type") == "disambiguation":
                return None

            extract = (summary_data.get("extract") or "").strip()
            if not extract:
                return None

            chosen_title = str(summary_data.get("title") or title)
            if not self._looks_relevant_internet_match(query, chosen_title, extract):
                logger.info("Rejected internet result due to low topical relevance: query=%s title=%s", query, chosen_title)
                return None

            sentences = re.split(r"(?<=[.!?])\s+", extract)
            concise = " ".join(sentences[:2]).strip()
            return concise or extract[:320]
        except Exception as exc:
            logger.info("Internet enrichment failed: %s", exc)
            return None

    def _persist_internet_knowledge(self, user_input: str, response: str) -> bool:
        """Persist internet-learned response into intents JSON for future local matches."""
        intents_path = self.kb_matcher.intents_path
        if not intents_path.exists():
            return False

        normalized = self.kb_matcher._normalize_text(user_input)
        raw_pattern = user_input.strip()
        tag = self._slugify_tag(normalized)

        try:
            with _KB_WRITE_LOCK:
                with open(intents_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                intents = data.setdefault("intents", [])
                target = None
                for intent in intents:
                    if intent.get("tag") == tag:
                        target = intent
                        break

                if target is None:
                    target = {
                        "tag": tag,
                        "patterns": [raw_pattern, normalized],
                        "responses": [response],
                    }
                    intents.append(target)
                else:
                    patterns = target.setdefault("patterns", [])
                    responses = target.setdefault("responses", [])
                    for p in [raw_pattern, normalized]:
                        if p and p not in patterns:
                            patterns.append(p)
                    if response not in responses:
                        responses.append(response)

                with open(intents_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            logger.warning("Failed to persist internet knowledge: %s", exc)
            return False

    def _graceful_fallback_response(self, user_input: str, user_context: Optional[Dict]) -> str:
        text = user_input.lower()

        if any(greet in text for greet in ["hi", "hello", "hey"]):
            return "Hi. I can still help with your built-in assistant features while I reconnect." \
                   " You can ask about tasks, reminders, or basic assistance."

        if any(word in text for word in ["task", "remind", "schedule", "todo"]):
            return "I can still help you organize tasks locally. Tell me what task you want to create and when."

        return "I am temporarily unable to generate an advanced response right now, but I am still here to help." \
               " Please try again in a moment."

    def _domain_bootstrap_response(self, user_input: str) -> Optional[str]:
        text = user_input.lower()

        if any(k in text for k in ["what is llm", "what are llms", "large language model", "llm"]):
            return (
                "An LLM (Large Language Model) is an AI model trained on very large text datasets to understand and generate natural language. "
                "It works by predicting likely next tokens based on context, which allows it to answer questions, summarize text, write code, and assist in conversation. "
                "In practical terms, LLM quality depends on training data, model size, and alignment methods. Even strong LLMs can make mistakes, so important facts should be verified."
            )

        if any(k in text for k in ["super computer", "supercomputer"]):
            return (
                "A supercomputer works by splitting very large problems into many smaller tasks and running them in parallel across thousands of processors. "
                "It uses high-speed interconnects so compute nodes can exchange data quickly, and advanced storage systems to stream huge datasets without bottlenecks. "
                "This architecture is used for weather forecasting, scientific simulation, molecular research, and large-scale AI workloads where normal computers are too slow."
            )

        if any(k in text for k in ["stomack ache", "stomach ache", "stomach pain", "my stomach hurts"]):
            return (
                "Stomach ache can occur due to indigestion, gas, food infection, acidity, constipation, or stress. "
                "For mild symptoms, rest, hydrate, and prefer light food while avoiding spicy or oily meals for a few hours. "
                "Please seek medical care promptly if pain is severe, persistent, or includes fever, vomiting, blood in stool/vomit, chest pain, or dehydration. "
                "This is general guidance and not a diagnosis."
            )

        if any(k in text for k in ["richest man in india", "richest person in india", "wealthiest in india"]):
            return (
                "In most recent rankings, Mukesh Ambani is commonly listed as the richest person in India, with Gautam Adani often close behind. "
                "These rankings can change quickly based on stock market movement, company valuations, and currency changes. "
                "For day-accurate figures, cross-check with live billionaire trackers like Forbes or Bloomberg."
            )

        if "polymorphism" in text and ("oops" in text or "oop" in text or "object" in text):
            return (
                "Polymorphism in OOP means one interface, many implementations. "
                "In practice, the same method name can behave differently depending on the object type. "
                "Common forms are method overriding (runtime polymorphism) and method overloading (compile-time polymorphism, language-dependent). "
                "This improves code flexibility and reuse because you can write generic code that works with multiple object types."
            )

        if any(k in text for k in ["benefits of exercise", "benefit of exercise", "exercise benefits"]):
            return (
                "Regular exercise improves heart health, muscle strength, metabolism, sleep quality, and mental well-being. "
                "It can reduce risk for diabetes, obesity, and hypertension, while also lowering stress and improving mood. "
                "A practical plan is 150 minutes of moderate activity per week plus 2 days of strength training."
            )

        return None

    def get_response(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict]] = None,
        user_context: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, object]:
        # Stage 1: local KB/classifier first.
        local = self.search_local_knowledge_base(user_input)
        if local:
            return local

        # Stage 2: internet enrichment for unknown queries, then persist into local KB.
        internet_response = self._fetch_internet_knowledge(user_input)
        if internet_response:
            self._persist_internet_knowledge(user_input, internet_response)
            self.refresh_knowledge_base()
            return {
                "response": internet_response,
                "source": "internet_knowledge",
                "used_api": False,
                "intent": self._slugify_tag(self.kb_matcher._normalize_text(user_input)),
                "confidence": 0.7,
            }

        # Stage 2.5: domain bootstrap response when local/internet are unavailable.
        bootstrap_response = self._domain_bootstrap_response(user_input)
        if bootstrap_response:
            return {
                "response": bootstrap_response,
                "source": "domain_bootstrap",
                "used_api": False,
                "intent": None,
                "confidence": 0.65,
            }

        # Stage 3: external LLM only if local and internet sources do not match.
        try:
            llm_response, provider = self.llm_client.generate(
                user_input=user_input,
                history=conversation_history,
                system_prompt=system_prompt,
            )
            return {
                "response": llm_response,
                "source": f"{provider}_llm",
                "used_api": True,
                "intent": None,
                "confidence": None,
            }
        except Exception:
            # Stage 4: graceful fallback. Never expose provider internals to users.
            return {
                "response": self._graceful_fallback_response(user_input, user_context),
                "source": "graceful_fallback",
                "used_api": False,
                "intent": None,
                "confidence": None,
            }

    def retrieve_context(self, query: str) -> List[Dict[str, str]]:
        """
        Placeholder retrieval interface for future RAG integration.
        Return shape is designed for chunk-based retrieval backends.
        """
        _ = query
        return []
