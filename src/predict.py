"""Sentiment prediction module."""

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_processing import load_config, text_preprocessor
from src.logging_config import get_logger

logger = get_logger(__name__)


class SentimentPredictor:
    """
    Sentiment predictor that loads a trained model and performs inference.
    
    Attributes:
        device: PyTorch device (cuda/cpu)
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model for sequence classification
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(self, config_path: str | None = None, model_path: str | None = None):
        """
        Initialize the predictor.
        
        Args:
            config_path: Path to configuration file. Uses default if None.
            model_path: Direct path to model. Overrides config if provided.
        """
        config = load_config(config_path)
        self.max_length = config["train"]["max_length"]
        
        if model_path is None:
            model_path = config["model"]["save_path"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_path} on device {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("SentimentPredictor initialized successfully")

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze.
        
        Returns:
            Dictionary with text, sentiment label, and label_id.
        """
        processed_text = text_preprocessor(text)
        
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_class_id = logits.argmax().item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()
        sentiment = "HAPPY" if predicted_class_id == 0 else "SAD"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "label_id": predicted_class_id,
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts.
        
        Returns:
            List of prediction dictionaries.
        """
        if not texts:
            return []
        
        processed_texts = [text_preprocessor(t) for t in texts]
        
        inputs = self.tokenizer(
            processed_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = logits.argmax(dim=-1)
        
        results = []
        for i, (text, pred_id) in enumerate(zip(texts, predicted_ids)):
            pred_id_item = pred_id.item()
            sentiment = "HAPPY" if pred_id_item == 0 else "SAD"
            confidence = probs[i][pred_id_item].item()
            
            results.append({
                "text": text,
                "sentiment": sentiment,
                "label_id": pred_id_item,
                "confidence": round(confidence, 4),
            })
        
        return results