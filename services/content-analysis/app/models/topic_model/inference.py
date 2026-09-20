from pathlib import Path
import json
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BinXClassifier:
    def __init__(self, bundle_dir, device=None):
        self.root = Path(bundle_dir)
        self.config = json.loads((self.root / "ensemble_config.json").read_text(encoding="utf-8"))
        label_info = json.loads((self.root / "label_map.json").read_text(encoding="utf-8"))
        self.labels = label_info["labels"]
        threshold_map = json.loads((self.root / "thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = np.array([threshold_map[label] for label in self.labels], dtype=float)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.components = []
        for spec in self.config["components"]:
            path = self.root / spec["path"]
            if spec["type"] == "tfidf":
                loaded = joblib.load(path)
                self.components.append((spec, loaded))
            else:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
                self.components.append((spec, (tokenizer, model)))

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -40, 40)
        return 1.0 / (1.0 + np.exp(-values))

    def predict(self, text):
        text = str(text).strip()
        combined = np.zeros(len(self.labels), dtype=np.float32)
        for spec, loaded in self.components:
            if spec["type"] == "tfidf":
                features = loaded["features"].transform([text])
                probs = loaded["classifier"].predict_proba(features)[0]
            else:
                tokenizer, model = loaded
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=spec["max_length"])
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded).logits[0].float().cpu().numpy()
                probs = self._sigmoid(logits)
            combined += float(spec["weight"]) * probs

        selected = np.flatnonzero(combined >= self.thresholds)
        selected = selected[np.argsort(combined[selected])[::-1]]
        selected = selected[: int(self.config.get("max_returned_topics", 6))]
        if len(selected) == 0:
            return {
                "classification_status": "unclassified",
                "primary_topics": [],
                "secondary_topics": [],
                "topic_count": 0,
                "reason_code": "NO_TOPIC_ABOVE_THRESHOLD",
                "all_scores": {label: round(float(combined[i]), 6) for i, label in enumerate(self.labels)},
            }

        top_score = float(combined[selected[0]])
        max_primary = int(self.config.get("max_primary_topics", 3))
        margin = float(self.config.get("primary_score_margin", 0.10))
        primary_ids = [int(i) for i in selected if top_score - float(combined[i]) <= margin][:max_primary]
        primary_set = set(primary_ids)
        secondary_ids = [int(i) for i in selected if int(i) not in primary_set]

        def topic_item(i):
            return {"topic": self.labels[i], "confidence": round(float(combined[i]), 6)}

        return {
            "classification_status": "classified",
            "primary_topics": [topic_item(i) for i in primary_ids],
            "secondary_topics": [topic_item(i) for i in secondary_ids],
            "topic_count": int(len(selected)),
            "all_scores": {label: round(float(combined[i]), 6) for i, label in enumerate(self.labels)},
        }
