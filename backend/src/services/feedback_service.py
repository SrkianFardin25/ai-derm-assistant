"""Feedback aggregation for model-monitoring signals."""

from collections import Counter
from threading import RLock
from typing import Dict, Optional


class FeedbackService:
    """Aggregate user feedback in-process; durable persistence remains optional."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counts: Counter[str] = Counter()
        self._class_pairs: Counter[str] = Counter()

    def record(
        self,
        feedback: str,
        predicted_class: str,
        corrected_class: Optional[str] = None,
    ) -> Dict[str, int | str]:
        feedback = feedback.upper().strip()
        if feedback not in {"CORRECT", "INCORRECT", "UNCERTAIN"}:
            raise ValueError("feedback must be CORRECT, INCORRECT, or UNCERTAIN")
        with self._lock:
            self._counts[feedback] += 1
            if feedback == "INCORRECT" and corrected_class:
                self._class_pairs[f"{predicted_class}->{corrected_class}"] += 1
            return self.summary()

    def summary(self) -> Dict[str, object]:
        with self._lock:
            return {
                "counts": dict(self._counts),
                "incorrect_class_pairs": dict(self._class_pairs),
                "total": sum(self._counts.values()),
            }
