import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models"

TOPIC_DIR = MODELS_DIR / "topic_model"
SAFETY_DIR = MODELS_DIR / "safety_model"
DIFFICULTY_DIR = MODELS_DIR / "difficulty_model"


class ModelService:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading BinX models on: {self.device}")

        self._load_topic_model()
        self._load_safety_model()
        self._load_difficulty_model()

        print("All BinX models loaded successfully.")

    def _load_topic_model(self):
        inference_path = TOPIC_DIR / "inference.py"

        if not inference_path.exists():
            raise FileNotFoundError(
                f"Topic inference file missing: {inference_path}"
            )

        module_spec = importlib.util.spec_from_file_location(
            "binx_topic_inference",
            inference_path,
        )

        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                "Could not load topic inference module."
            )

        topic_module = importlib.util.module_from_spec(
            module_spec
        )
        module_spec.loader.exec_module(topic_module)

        self.topic_classifier = topic_module.BinXClassifier(
            TOPIC_DIR,
            device=self.device,
        )

    def _load_safety_model(self):
        self.safety_tokenizer = (
            AutoTokenizer.from_pretrained(
                SAFETY_DIR,
                local_files_only=True,
            )
        )

        self.safety_model = (
            AutoModelForSequenceClassification.from_pretrained(
                SAFETY_DIR,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )

        labels_path = SAFETY_DIR / "labels.json"
        thresholds_path = SAFETY_DIR / "thresholds.json"

        self.safety_labels = json.loads(
            labels_path.read_text(encoding="utf-8")
        )

        self.safety_thresholds = json.loads(
            thresholds_path.read_text(encoding="utf-8")
        )

    def _load_difficulty_model(self):
        self.difficulty_tokenizer = (
            AutoTokenizer.from_pretrained(
                DIFFICULTY_DIR,
                local_files_only=True,
            )
        )

        self.difficulty_model = (
            AutoModelForSequenceClassification.from_pretrained(
                DIFFICULTY_DIR,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )

        label_map_path = DIFFICULTY_DIR / "label_map.json"

        if label_map_path.exists():
            label_map = json.loads(
                label_map_path.read_text(encoding="utf-8")
            )

            self.difficulty_id2label = {
                int(key): value
                for key, value in label_map["id2label"].items()
            }
        else:
            self.difficulty_id2label = {
                int(key): value
                for key, value
                in self.difficulty_model.config.id2label.items()
            }

    def predict_topic(self, text: str) -> dict[str, Any]:
        return self.topic_classifier.predict(text)

    def predict_safety(self, text: str) -> dict[str, Any]:
        encoded = self.safety_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = self.safety_model(
                **encoded
            ).logits[0]

            probabilities = torch.sigmoid(
                logits
            ).detach().cpu().tolist()

        all_scores = {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(
                self.safety_labels
            )
        }

        risks = []

        for label in self.safety_labels:
            probability = all_scores[label]
            threshold = float(
                self.safety_thresholds[label]
            )

            if probability >= threshold:
                risks.append(
                    {
                        "category": label,
                        "confidence": probability,
                        "threshold": threshold,
                    }
                )

        risks.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        review_required = len(risks) > 0

        return {
            "safety_status": (
                "REVIEW_REQUIRED"
                if review_required
                else "SAFE"
            ),
            "risk_categories": risks,
            "review_required": review_required,
            "recommendation_signal": (
                "DOWNRANK_OR_HOLD"
                if review_required
                else "ALLOW"
            ),
            "all_scores": all_scores,
        }

    def predict_difficulty(
        self,
        text: str,
    ) -> dict[str, Any]:
        encoded = self.difficulty_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = self.difficulty_model(
                **encoded
            ).logits[0]

            probabilities = torch.softmax(
                logits,
                dim=-1,
            ).detach().cpu().tolist()

        predicted_id = int(
            max(
                range(len(probabilities)),
                key=lambda index: probabilities[index],
            )
        )

        predicted_label = (
            self.difficulty_id2label[predicted_id]
        )

        all_scores = {
            self.difficulty_id2label[index]: round(
                float(probability),
                6,
            )
            for index, probability
            in enumerate(probabilities)
        }

        return {
            "level": predicted_label,
            "confidence": round(
                float(probabilities[predicted_id]),
                6,
            ),
            "all_scores": all_scores,
        }

    def analyze(self, text: str) -> dict[str, Any]:
        text = str(text).strip()

        if not text:
            raise ValueError("Text cannot be empty.")

        topic_result = self.predict_topic(text)
        safety_result = self.predict_safety(text)
        difficulty_result = self.predict_difficulty(text)

        return {
            "topics": topic_result,
            "difficulty": difficulty_result,
            "safety": safety_result,
            "needs_review": safety_result[
                "review_required"
            ],
            "model_versions": {
                "topic": "binx-topic-v6",
                "safety": "binx-safety-v2",
                "difficulty": "binx-difficulty-v1",
            },
        }


_service_instance = None
_service_lock = threading.Lock()


def get_model_service() -> ModelService:
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ModelService()

    return _service_instance


def models_are_loaded() -> bool:
    return _service_instance is not None