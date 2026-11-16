"""Groq Clinical Advisory Service with Static Fallback."""

import os
from typing import Any, Dict, Optional
from src.domain.knowledge_cards import get_knowledge_card
from src.domain.models import PredictionResult

try:
    from groq import Groq
except ImportError:
    Groq = None


class GroqService:
    """Provides patient-friendly clinical educational summaries using Groq (Llama 3.3 70B)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = None
        if self.api_key and Groq is not None:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def __repr__(self) -> str:
        return f"<GroqService model='{self.model}' active={self.client is not None}>"

    def __str__(self) -> str:
        return self.__repr__()

    def generate_clinical_summary(
        self,
        prediction: PredictionResult,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate patient educational narrative, incorporating optional patient context if present."""
        card = get_knowledge_card(prediction.top_class)

        if not self.client:
            return self._build_static_summary(prediction, card, patient_metadata)

        system_instruction = (
            "You are a board-certified dermatologist writing a concise clinical chart note for a patient. "
            "Write in a natural, authoritative, physician-grade clinical voice. "
            "RULES: "
            "1. Strictly 2 to 3 sentences total (under 75 words). "
            "2. NEVER use markdown tables, bullet points, headers, asterisks, or pipe characters. "
            "3. NEVER use AI meta-intros, conversational preamble, or disclaimers (e.g. no 'As an AI', 'What the model saw'). "
            "4. Cover: 1) What the lesion is in plain terms, 2) Why its features indicate its risk tier, and 3) Recommended clinical step. "
            "5. Strictly respect the algorithmic classification and risk tier determined by the model; do not alter or override the diagnostic risk assessment."
        )

        patient_section = ""
        if patient_metadata:
            meta_items = []
            if patient_metadata.get("age") is not None:
                meta_items.append(f"- Reported Patient Age: {patient_metadata['age']} years")
            if patient_metadata.get("sex"):
                meta_items.append(f"- Reported Biological Sex: {str(patient_metadata['sex']).title()}")
            if patient_metadata.get("anatomical_site"):
                meta_items.append(f"- Reported Anatomical Site: {patient_metadata['anatomical_site']}")
            if patient_metadata.get("history_of_melanoma") is not None:
                hist_str = "Yes (Personal or familial history)" if patient_metadata["history_of_melanoma"] else "No prior history"
                meta_items.append(f"- History of Melanoma: {hist_str}")

            if meta_items:
                patient_section = (
                    "Patient-Reported Context (Self-Reported, Independent of Image Analysis):\n"
                    + "\n".join(meta_items)
                    + "\n\n"
                )

        prompt = (
            f"Model Algorithmic Assessment:\n"
            f"- Clinical Classification: {prediction.top_name} ({prediction.top_class})\n"
            f"- Risk Tier: {prediction.risk_level.value}\n"
            f"- Calibrated Confidence: {prediction.calibrated_confidence * 100:.1f}%\n"
            f"- Clinical Ground Truth: {card['description']}\n"
            f"- Recommended Action: {card['urgency']}\n\n"
            f"{patient_section}"
            f"Write the concise 2-to-3 sentence clinical note."
        )

        fallback_models = ["llama-3.1-8b-instant", "gemma2-9b-it"]
        for model_name in [self.model] + fallback_models:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=model_name,
                    temperature=0.1,
                    max_tokens=100
                )
                content = chat_completion.choices[0].message.content
                if content:
                    return content.strip()
            except Exception:
                continue
        return self._build_static_summary(prediction, card, patient_metadata)

    def _build_static_summary(
        self,
        prediction: PredictionResult,
        card: dict,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Concise physician-grade fallback narrative built from static clinical knowledge cards."""
        site_str = ""
        if patient_metadata and patient_metadata.get("anatomical_site"):
            site_str = f" on the {patient_metadata['anatomical_site']}"
        return (
            f"Clinical impression is consistent with {card['common_name']}{site_str}. Classification triage: {prediction.risk_level.value} with {prediction.calibrated_confidence * 100:.1f}% confidence. "
            f"{card['description'].split('.')[0]}. "
            f"{card['urgency']}"
        )
