"""
Production-level hybrid chatbot response system.

Flow:
1) Accept user input
2) Search local intents JSON
3) Fuzzy/similarity match to best intent
4) Return local response on match
5) Otherwise call external LLM (Gemini/OpenAI)
6) If API fails, return graceful fallback message
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ai_services.gemini_handler import ask_gemini
from ai_services.llm_handler import ask_llm

logger = logging.getLogger(__name__)


@dataclass
class IntentMatch:
    """Structured match output for local intent retrieval."""

    tag: str
    response: str
    score: float


class HybridChatbotEngine:
    """Hybrid engine that prefers local KB before external LLM usage."""

    def __init__(
        self,
        intents_path: Optional[str] = None,
        similarity_threshold: float = 0.72,
        enable_semantic_search: bool = False,
    ):
        self.intents_path = Path(intents_path) if intents_path else self._default_intents_path()
        self.similarity_threshold = similarity_threshold
        self.enable_semantic_search = enable_semantic_search

        self.intents_data: Dict = {"intents": []}
        self._intent_index: List[Tuple[str, str, List[str]]] = []

        self._rapidfuzz_process = None
        self._rapidfuzz_fuzz = None

        self._semantic_model = None
        self._semantic_pattern_texts: List[str] = []
        self._semantic_pattern_meta: List[Tuple[str, List[str]]] = []
        self._semantic_pattern_vectors = None

        self.load_intents()
        self._init_rapidfuzz()
        if self.enable_semantic_search:
            self._init_semantic_search()

    def _default_intents_path(self) -> Path:
        """Resolve best default intents file from known project paths."""
        base = Path(__file__).resolve().parent
        configured_path = os.getenv("CHATBOT_INTENTS_PATH", "").strip()
        if configured_path:
            candidate = Path(configured_path)
            if not candidate.is_absolute():
                candidate = (base.parent / candidate).resolve()
            if candidate.exists():
                return candidate

        candidates = [
            base / "intents_example.json",
            base / "neural_model" / "intents.json",
            base.parent / "Chatbot" / "chatbot_codes" / "intents.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _normalize_text(self, text: str) -> str:
        """Normalize user input and patterns for robust matching."""
        text = text.lower().strip()
        text = text.replace("what's", "what is")
        text = text.replace("whats", "what is")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def _init_rapidfuzz(self) -> None:
        """Optional fast fuzzy matcher; falls back to difflib if unavailable."""
        try:
            from rapidfuzz import fuzz, process

            self._rapidfuzz_process = process
            self._rapidfuzz_fuzz = fuzz
            logger.info("rapidfuzz enabled for intent matching")
        except Exception:
            logger.info("rapidfuzz not available; using difflib fallback")

    def _init_semantic_search(self) -> None:
        """Optional semantic retrieval with sentence-transformers for future RAG-readiness."""
        try:
            from sentence_transformers import SentenceTransformer

            self._semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._build_semantic_index()
            logger.info("semantic search enabled")
        except Exception as exc:
            logger.warning("semantic search unavailable: %s", exc)
            self._semantic_model = None

    def _build_semantic_index(self) -> None:
        if not self._semantic_model:
            return

        self._semantic_pattern_texts = []
        self._semantic_pattern_meta = []

        for intent in self.intents_data.get("intents", []):
            tag = intent.get("tag", "unknown")
            responses = intent.get("responses", [])
            for pattern in intent.get("patterns", []):
                if isinstance(pattern, str) and pattern.strip():
                    normalized = self._normalize_text(pattern)
                    self._semantic_pattern_texts.append(normalized)
                    self._semantic_pattern_meta.append((tag, responses))

        if not self._semantic_pattern_texts:
            return

        self._semantic_pattern_vectors = self._semantic_model.encode(
            self._semantic_pattern_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def load_intents(self) -> Dict:
        """
        Load intents from JSON and build in-memory index.

        Returns:
            Parsed intents dictionary.
        """
        if not self.intents_path.exists():
            logger.warning("intents file not found at %s", self.intents_path)
            self.intents_data = {"intents": []}
            self._intent_index = []
            return self.intents_data

        try:
            with open(self.intents_path, "r", encoding="utf-8") as file:
                self.intents_data = json.load(file)
        except Exception as exc:
            logger.exception("failed to load intents JSON: %s", exc)
            self.intents_data = {"intents": []}

        self._intent_index = []
        for intent in self.intents_data.get("intents", []):
            tag = intent.get("tag", "unknown")
            responses = intent.get("responses", [])
            for pattern in intent.get("patterns", []):
                if isinstance(pattern, str) and pattern.strip():
                    self._intent_index.append((self._normalize_text(pattern), tag, responses))

        if self.enable_semantic_search and self._semantic_model is not None:
            self._build_semantic_index()

        logger.info("loaded %s intent patterns", len(self._intent_index))
        return self.intents_data

    def _best_fuzzy_score_difflib(self, user_text: str) -> Tuple[float, Optional[Tuple[str, str, List[str]]]]:
        """Difflib fallback scoring when rapidfuzz is unavailable."""
        best_score = 0.0
        best_item = None

        for pattern, tag, responses in self._intent_index:
            ratio = SequenceMatcher(None, user_text, pattern).ratio()
            token_overlap = self._token_overlap(user_text, pattern)
            score = max(ratio, token_overlap)
            if score > best_score:
                best_score = score
                best_item = (pattern, tag, responses)

        return best_score, best_item

    def _best_fuzzy_score_rapidfuzz(self, user_text: str) -> Tuple[float, Optional[Tuple[str, str, List[str]]]]:
        """Rapidfuzz scoring with robust handling for phrase variations."""
        if not self._rapidfuzz_process or not self._rapidfuzz_fuzz or not self._intent_index:
            return 0.0, None

        patterns = [item[0] for item in self._intent_index]
        result = self._rapidfuzz_process.extractOne(
            query=user_text,
            choices=patterns,
            scorer=self._rapidfuzz_fuzz.WRatio,
        )
        if not result:
            return 0.0, None

        matched_pattern = result[0]
        score = float(result[1]) / 100.0

        for item in self._intent_index:
            if item[0] == matched_pattern:
                return score, item
        return 0.0, None

    def _token_overlap(self, user_text: str, pattern_text: str) -> float:
        """Token overlap helps short queries like 'name' map correctly."""
        user_tokens = set(re.findall(r"[a-z0-9]+", user_text))
        pattern_tokens = set(re.findall(r"[a-z0-9]+", pattern_text))
        if not user_tokens or not pattern_tokens:
            return 0.0
        overlap = len(user_tokens.intersection(pattern_tokens))
        return overlap / max(1, len(pattern_tokens))

    def _semantic_match(self, user_text: str) -> Optional[IntentMatch]:
        """Optional embedding-based semantic intent matching."""
        if not self._semantic_model or self._semantic_pattern_vectors is None:
            return None

        try:
            import numpy as np

            query_vector = self._semantic_model.encode([user_text], normalize_embeddings=True, show_progress_bar=False)[0]
            similarities = np.dot(self._semantic_pattern_vectors, query_vector)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score < self.similarity_threshold:
                return None

            tag, responses = self._semantic_pattern_meta[best_idx]
            if not responses:
                return None

            return IntentMatch(
                tag=tag,
                response=random.choice(responses),
                score=best_score,
            )
        except Exception as exc:
            logger.warning("semantic match failed: %s", exc)
            return None

    def find_best_intent(self, user_input: str) -> Optional[IntentMatch]:
        """
        Find the best matching intent using fuzzy matching and optional semantic search.

        Handles variants such as:
        - whats your name
        - what is your name
        - name
        """
        if not user_input or not user_input.strip() or not self._intent_index:
            return None

        normalized_input = self._normalize_text(user_input)
        query_tokens = set(re.findall(r"[a-z0-9]+", normalized_input))

        # Single-token query handling (e.g., "name") with exact token containment.
        if len(query_tokens) == 1:
            token = next(iter(query_tokens))
            token_candidates: List[Tuple[str, str, List[str]]] = []
            for pattern, tag, responses in self._intent_index:
                pattern_tokens = set(re.findall(r"[a-z0-9]+", pattern))
                if token in pattern_tokens and responses:
                    token_candidates.append((pattern, tag, responses))

            if token_candidates:
                _, tag, responses = random.choice(token_candidates)
                return IntentMatch(tag=tag, response=random.choice(responses), score=0.9)

        # First pass: exact phrase checks and safe multi-token containment.
        for pattern, tag, responses in self._intent_index:
            pattern_tokens = set(re.findall(r"[a-z0-9]+", pattern))
            contains_match = (
                (len(query_tokens) >= 2 and normalized_input in pattern)
                or (len(pattern_tokens) >= 2 and pattern in normalized_input)
            )
            if normalized_input == pattern or contains_match:
                if responses:
                    return IntentMatch(tag=tag, response=random.choice(responses), score=0.96)

        # Second pass: fuzzy best match.
        if self._rapidfuzz_process is not None:
            best_score, best_item = self._best_fuzzy_score_rapidfuzz(normalized_input)
        else:
            best_score, best_item = self._best_fuzzy_score_difflib(normalized_input)

        if best_item and best_score >= self.similarity_threshold:
            _, tag, responses = best_item
            if responses:
                return IntentMatch(tag=tag, response=random.choice(responses), score=best_score)

        # Third pass: optional semantic search.
        semantic_match = self._semantic_match(normalized_input)
        if semantic_match:
            return semantic_match

        return None

    def call_llm_api(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """
        Call external LLM as fallback only when local intent match fails.

        Provider order:
        1) configured provider (gemini/openai)
        2) secondary provider failover

        Never expose raw API failures to end users.
        """
        provider_name = (provider or os.getenv("PRIMARY_LLM_PROVIDER") or "gemini").lower().strip()
        if provider_name not in {"gemini", "openai"}:
            provider_name = "gemini"

        ordered_providers = [provider_name, "openai" if provider_name == "gemini" else "gemini"]

        for p in ordered_providers:
            try:
                if p == "gemini":
                    result = ask_gemini(user_input=user_input, history=history or [], system_prompt=system_prompt)
                else:
                    result = ask_llm(user_input=user_input, history=history or [], system_prompt=system_prompt)

                # Handler modules already return sanitized responses; still guard unsafe/empty output.
                if result and isinstance(result, str) and result.strip():
                    return result.strip()
            except Exception as exc:
                logger.warning("LLM provider %s failed: %s", p, exc)

        # Graceful fallback after all providers fail.
        fallback_candidates = [
            "Sorry, I did not understand that clearly.",
            "Can you rephrase that?",
            "I am having a temporary issue answering that. Please try again.",
        ]
        return random.choice(fallback_candidates)

    def get_response(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Main response entrypoint for production usage.

        Returns:
            dict with response text, source, and metadata.
        """
        # Step 1-4: local knowledge base first.
        intent_match = self.find_best_intent(user_input)
        if intent_match:
            return {
                "response": intent_match.response,
                "source": "local_intent",
                "tag": intent_match.tag,
                "score": round(intent_match.score, 4),
                "used_api": False,
            }

        # Step 5-6: external LLM with graceful fallback.
        llm_response = self.call_llm_api(
            user_input=user_input,
            history=history,
            system_prompt=system_prompt,
            provider=provider,
        )

        return {
            "response": llm_response,
            "source": "external_llm_or_fallback",
            "tag": None,
            "score": None,
            "used_api": llm_response not in {
                "Sorry, I did not understand that clearly.",
                "Can you rephrase that?",
                "I am having a temporary issue answering that. Please try again.",
            },
        }

    def retrieve_context(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieval extension point for future RAG integration.
        Replace with vector DB or document retriever in production.
        """
        _ = query
        return []


# Shared default engine for simple function-style usage.
_DEFAULT_ENGINE = HybridChatbotEngine()


def load_intents(intents_path: Optional[str] = None) -> Dict:
    """Load intents from JSON file."""
    if intents_path:
        engine = HybridChatbotEngine(intents_path=intents_path)
        return engine.load_intents()
    return _DEFAULT_ENGINE.load_intents()


def find_best_intent(user_input: str) -> Optional[Dict[str, object]]:
    """Find the best matching intent for a user message."""
    match = _DEFAULT_ENGINE.find_best_intent(user_input)
    if not match:
        return None
    return {
        "tag": match.tag,
        "response": match.response,
        "score": round(match.score, 4),
    }


def call_llm_api(
    user_input: str,
    history: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Call external LLM provider with graceful fallback handling."""
    return _DEFAULT_ENGINE.call_llm_api(
        user_input=user_input,
        history=history,
        system_prompt=system_prompt,
        provider=provider,
    )


def get_response(
    user_input: str,
    history: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, object]:
    """Primary function for chatbot response generation."""
    return _DEFAULT_ENGINE.get_response(
        user_input=user_input,
        history=history,
        system_prompt=system_prompt,
        provider=provider,
    )
