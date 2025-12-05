"""Data processing utilities for sentiment analysis."""

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from hazm import Normalizer

from src.logging_config import get_logger

logger = get_logger(__name__)

# Initialize normalizer once at module load (hazm Normalizer is expensive)
_normalizer = Normalizer()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load parameters from the configuration file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "params.yml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_and_clean_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load raw data and apply all cleaning steps.
    
    Args:
        config: Configuration dictionary containing data paths and preprocessing params.
    
    Returns:
        Cleaned DataFrame with 'comment' and 'label_id' columns.
    """
    raw_path = get_project_root() / config["data"]["raw_path"]
    logger.info(f"Loading data from {raw_path}")
    
    df = pd.read_csv(raw_path, on_bad_lines="skip", delimiter=",")
    initial_size = len(df)

    # Remove unnecessary columns
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    
    # Remove nulls and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["comment"], keep="first", inplace=True)

    # Normalize labels and fix inconsistencies
    df["label"] = df["label"].str.upper().str.strip()
    df = df[~((df["label"] == "HAPPY") & (df["label_id"] != 0))]
    df = df[~((df["label"] == "SAD") & (df["label_id"] != 1))]
    df["label_id"] = df["label_id"].astype(int)

    # Filter non-Persian and long comments
    df = df[~df["comment"].str.contains(r"[a-zA-Z]", na=False)]
    df["word_count"] = df["comment"].apply(lambda x: len(str(x).split()))
    max_words = df["word_count"].quantile(config["preprocess"]["max_words_quantile"])
    df = df[df["word_count"] <= max_words]

    # Reset index and select relevant columns
    df = df[["comment", "label_id"]].reset_index(drop=True)
    
    logger.info(
        f"Data loaded and cleaned. Original: {initial_size}, Final: {len(df)} "
        f"({len(df) / initial_size * 100:.1f}% retained)"
    )
    return df


def text_preprocessor(text: str) -> str:
    """
    Preprocess text for BERT model.
    
    Normalizes Persian text and removes unwanted characters.
    No stopword removal or stemming (BERT handles context).
    
    Args:
        text: Raw input text.
    
    Returns:
        Preprocessed text.
    """
    # Normalize using hazm
    text = _normalizer.normalize(text)
    # Remove anything that's not Persian characters, numbers, or whitespace
    text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
    return text


def preprocess_batch(texts: list[str]) -> list[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of raw input texts.
    
    Returns:
        List of preprocessed texts.
    """
    return [text_preprocessor(t) for t in texts]