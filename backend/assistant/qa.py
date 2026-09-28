"""
Q&A Assistant — Stage 5 (Local Assistant), optional

Answers questions about the experiment procedure using:
  1. KeywordQA    — simple keyword/TF-IDF search over SOP text (always works)
  2. OllamaQA     — local LLM via Ollama (upgrade path, needs Ollama running)

Example queries: "what's the next step?", "what objects do I need?",
"what happens after I open the red box?"
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseQA(ABC):
    @abstractmethod
    def answer(self, question: str, context: str = "") -> str:
        """Answer a question, optionally with extra context."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class KeywordQA(BaseQA):
    """
    Simple keyword-matching Q&A over the SOP text.
    Zero dependencies, always works.  Handles common questions about
    the experiment procedure by pattern matching.
    """

    def __init__(self, sop_text: str, step_descriptions: list[dict] | None = None):
        self._sop_text = sop_text.lower()
        self._sop_text_original = sop_text
        self._steps = step_descriptions or []

    def answer(self, question: str, context: str = "") -> str:
        q = question.lower().strip()

        # "what's next?" / "next step"
        if any(kw in q for kw in ["next", "what's next", "whats next", "after this"]):
            return self._answer_next_step(context)

        # "what step am I on?" / "current step"
        if any(kw in q for kw in ["current", "which step", "what step", "where am i"]):
            return self._answer_current_step(context)

        # "what objects" / "what do I need"
        if any(kw in q for kw in ["object", "need", "require", "tool", "item"]):
            return self._answer_objects(context)

        # "how many steps" / "total"
        if any(kw in q for kw in ["how many", "total", "count"]):
            return f"The experiment has {len(self._steps)} steps in total."

        # "list steps" / "all steps" / "procedure"
        if any(kw in q for kw in ["list", "all step", "procedure", "steps"]):
            return self._answer_list_steps()

        # "help"
        if "help" in q:
            return (
                "I can answer questions about the experiment procedure. "
                "Try asking: 'What's next?', 'What objects do I need?', "
                "'List all steps', or 'How many steps?'"
            )

        # Generic: search SOP text for relevant sentences
        return self._search_sop(question)

    def _answer_next_step(self, context: str) -> str:
        # Try to parse current step from context
        current_id = self._parse_current_step_id(context)
        if current_id is not None and current_id < len(self._steps):
            next_step = self._steps[current_id]  # 0-indexed, current_id is 1-indexed
            return f"The next step is Step {next_step['step_id']}: {next_step['description']}."
        elif self._steps:
            return f"The first step is: {self._steps[0]['description']}."
        return "I'm not sure what the next step is."

    def _answer_current_step(self, context: str) -> str:
        current_id = self._parse_current_step_id(context)
        if current_id is not None and 0 < current_id <= len(self._steps):
            step = self._steps[current_id - 1]
            return f"You are on Step {step['step_id']}: {step['description']}."
        return "I'm not sure which step you're on."

    def _answer_objects(self, context: str) -> str:
        current_id = self._parse_current_step_id(context)
        if current_id is not None and 0 < current_id <= len(self._steps):
            step = self._steps[current_id - 1]
            objects = step.get("required_objects", [])
            return f"For Step {step['step_id']}, you need: {', '.join(objects)}."

        # List all unique objects
        all_objects = set()
        for step in self._steps:
            all_objects.update(step.get("required_objects", []))
        return f"The experiment uses these objects: {', '.join(sorted(all_objects))}."

    def _answer_list_steps(self) -> str:
        if not self._steps:
            return "No steps loaded."
        lines = ["Here are all the steps:"]
        for step in self._steps:
            lines.append(f"  {step['step_id']}. {step['description']}")
        return "\n".join(lines)

    def _search_sop(self, question: str) -> str:
        """Fallback: find the most relevant sentence in the SOP text."""
        words = set(re.findall(r'\w+', question.lower()))
        words -= {"what", "is", "the", "a", "an", "how", "do", "does", "can", "i", "my", "to"}

        sentences = self._sop_text_original.split('\n')
        best_score = 0
        best_sentence = ""

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for w in words if w in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

        if best_sentence and best_score > 0:
            return best_sentence
        return "I'm not sure how to answer that. Try asking about the next step, required objects, or the procedure."

    def _parse_current_step_id(self, context: str) -> Optional[int]:
        """Try to extract current step ID from context string."""
        if not context:
            return None
        match = re.search(r'current_step[_\s]*(?:id)?[:\s]*(\d+)', context, re.I)
        if match:
            return int(match.group(1))
        match = re.search(r'step[:\s]*(\d+)', context, re.I)
        if match:
            return int(match.group(1))
        return None

    def is_available(self) -> bool:
        return True


class OllamaQA(BaseQA):
    """
    Local LLM Q&A via Ollama.
    Falls back to KeywordQA if Ollama isn't running or installed.
    """

    def __init__(self, model: str = "qwen2.5:7b-instruct-q4_0",
                 sop_text: str = "", step_descriptions: list[dict] | None = None):
        self._model = model
        self._sop_text = sop_text
        self._client = None
        self._fallback = KeywordQA(sop_text, step_descriptions)

        try:
            import ollama
            # Test connection
            ollama.list()
            self._client = ollama
            logger.info("Ollama Q&A initialized with model: %s", model)
        except Exception as e:
            logger.warning("Ollama not available (%s) — falling back to keyword Q&A", e)

    def answer(self, question: str, context: str = "") -> str:
        if not self._client:
            return self._fallback.answer(question, context)

        prompt = (
            f"You are an AI assistant helping an astronaut perform a scientific experiment. "
            f"Answer the question based on this procedure:\n\n{self._sop_text}\n\n"
            f"Current context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Give a brief, direct answer in 1-2 sentences."
        )

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error("Ollama error: %s — falling back to keyword", e)
            return self._fallback.answer(question, context)

    def is_available(self) -> bool:
        return self._client is not None


def create_qa(mode: str = "keyword", sop_text: str = "",
              step_descriptions: list[dict] | None = None, **kwargs) -> BaseQA:
    """Factory function for Q&A assistant."""
    if mode == "ollama":
        return OllamaQA(sop_text=sop_text, step_descriptions=step_descriptions, **kwargs)
    else:
        return KeywordQA(sop_text, step_descriptions)
