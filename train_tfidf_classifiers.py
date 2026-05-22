#!/usr/bin/env python3
"""Train TF-IDF sentiment classifiers on Nepali cleaned dataset.

This script performs the full pipeline:
1. Load cleaned_dataset.csv
2. Map labels 1.0/2.0 to POSITIVE/NEGATIVE
3. Final Nepali text preprocessing
4. Stratified train/validation/test split
5. TF-IDF vectorization
6. Train LogisticRegression, RandomForest, and XGBoost
7. Evaluate on validation and select best model
8. Test the selected model once on the test set
9. Save the trained vectorizer, best model, and label encoder
"""

import os
import re
import unicodedata
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required for this script. Install it with `pip install xgboost`."
    ) from exc


DATA_PATH = "cleaned_dataset.csv"
SAVE_DIR = "best_ml_model"
VECTORIZER_PATH = os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl")
COUNT_VECTORIZER_PATH = os.path.join(SAVE_DIR, "count_vectorizer.pkl")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "classifier.pkl")
LABEL_MAP_PATH = os.path.join(SAVE_DIR, "label_map.json")


@dataclass
class Config:
    random_state: int = 42
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    token_pattern: str = r"[\u0900-\u097F]{2,}"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 50000
    sublinear_tf: bool = True
    min_df: int = 2
    max_df: float = 0.95
    rf_n_estimators: int = 300
    rf_min_samples_leaf: int = 2
    lr_max_iter: int = 1000
    xgb_n_estimators: int = 300
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected `text` and `label` columns in cleaned_dataset.csv")

    # Cast text to string as a safety measure and preserve original values.
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(float)

    label_map = {1.0: "POSITIVE", 2.0: "NEGATIVE"}
    df["label"] = df["label"].map(label_map)

    if df["label"].isna().any():
        missing = df.loc[df["label"].isna(), :].head(5)
        raise ValueError(
            "Found unexpected label values in cleaned_dataset.csv. "
            f"Expected only 1.0 and 2.0. Sample rows:\n{missing}"
        )

    return df


def preprocess_text(series: pd.Series) -> pd.Series:
    url_pattern = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+", flags=re.IGNORECASE)
    mention_pattern = re.compile(r"@\S+")
    hashtag_pattern = re.compile(r"#\S+")
    devanagari_only_pattern = re.compile(r"[^\u0900-\u097F\s]+")
    whitespace_pattern = re.compile(r"\s+")

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize("NFC", text)
        text = url_pattern.sub(" ", text)
        text = mention_pattern.sub(" ", text)
        text = hashtag_pattern.sub(" ", text)
        text = devanagari_only_pattern.sub(" ", text)
        text = whitespace_pattern.sub(" ", text).strip()
        return text

    return series.map(clean_text)


def split_data(df: pd.DataFrame, cfg: Config):
    stratify_col = df["label"]

    train_temp, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_col,
    )

    train_df, val_df = train_test_split(
        train_temp,
        test_size=cfg.val_size / (cfg.train_size + cfg.val_size),
        random_state=cfg.random_state,
        stratify=train_temp["label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_vectorizers(cfg: Config) -> tuple[CountVectorizer, TfidfVectorizer]:
    count_vec = CountVectorizer(
        token_pattern=cfg.token_pattern,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )
    tfidf_vec = TfidfVectorizer(
        token_pattern=cfg.token_pattern,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )
    return count_vec, tfidf_vec


def train_models(X_train, y_train, scale_pos_weight: float, cfg: Config):
    models = {
        "LogisticRegression": LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=cfg.lr_max_iter,
            class_weight="balanced",
            random_state=cfg.random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            class_weight="balanced",
            min_samples_leaf=cfg.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=cfg.random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_learning_rate,
            max_depth=cfg.xgb_max_depth,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"  {name} training complete")

    return models


def evaluate_model(model, X, y_true, label_encoder: LabelEncoder):
    y_pred = model.predict(X)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": report,
        "y_pred": y_pred,
    }


def print_evaluation(name: str, metrics: dict):
    print(f"\n{name} evaluation")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(metrics["report"])


def save_artifacts(count_vec, tfidf_vec, model, label_encoder):
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(count_vec, COUNT_VECTORIZER_PATH)
    joblib.dump(tfidf_vec, VECTORIZER_PATH)
    joblib.dump(model, BEST_MODEL_PATH)
    label_map = {str(i): cls for i, cls in enumerate(label_encoder.classes_)}
    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False)
    print(f"\nSaved to {SAVE_DIR}/")
    print(f"   count_vectorizer.pkl")
    print(f"   tfidf_vectorizer.pkl")
    print(f"   classifier.pkl")
    print(f"   label_map.json — {label_map}")


def main():
    cfg = Config()
    df = load_dataset(DATA_PATH)

    print(f"Loaded {len(df):,} rows from {DATA_PATH}")
    print("Label distribution before preprocessing:")
    print(df["label"].value_counts())

    df["text"] = preprocess_text(df["text"])
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

    print(f"\nAfter final preprocessing: {len(df):,} rows remain")
    print("Label distribution after preprocessing:")
    print(df["label"].value_counts())

    label_encoder = LabelEncoder()
    df["label_enc"] = label_encoder.fit_transform(df["label"])

    if list(label_encoder.classes_) != ["NEGATIVE", "POSITIVE"]:
        raise ValueError(
            "Label encoder classes are not ordered as [NEGATIVE, POSITIVE]. "
            f"Got {label_encoder.classes_}"
        )

    X_train_df, X_val_df, X_test_df = split_data(df, cfg)
    print(f"\nTrain / val / test sizes: {len(X_train_df):,} / {len(X_val_df):,} / {len(X_test_df):,}")
    print("Train label distribution:")
    print(X_train_df["label"].value_counts())
    print("Val label distribution:")
    print(X_val_df["label"].value_counts())
    print("Test label distribution:")
    print(X_test_df["label"].value_counts())

    count_vec, tfidf_vec = build_vectorizers(cfg)
    X_train_count = count_vec.fit_transform(X_train_df["text"])
    X_train_tfidf = tfidf_vec.fit_transform(X_train_df["text"])
    X_train = hstack([X_train_count, X_train_tfidf])
    
    X_val_count = count_vec.transform(X_val_df["text"])
    X_val_tfidf = tfidf_vec.transform(X_val_df["text"])
    X_val = hstack([X_val_count, X_val_tfidf])
    
    X_test_count = count_vec.transform(X_test_df["text"])
    X_test_tfidf = tfidf_vec.transform(X_test_df["text"])
    X_test = hstack([X_test_count, X_test_tfidf])

    y_train = X_train_df["label_enc"].values
    y_val = X_val_df["label_enc"].values
    y_test = X_test_df["label_enc"].values

    counts = np.bincount(y_train)
    if len(counts) != 2:
        raise ValueError("Expected exactly two classes after encoding")
    neg_count, pos_count = counts[0], counts[1]
    scale_pos_weight = neg_count / pos_count
    print(f"\nTrain positive/negative counts: {pos_count:,} / {neg_count:,}")
    print(f"Computed XGBoost scale_pos_weight = {scale_pos_weight:.4f}")

    models = train_models(X_train, y_train, scale_pos_weight, cfg)

    validation_results = {}
    for name, model in models.items():
        validation_results[name] = evaluate_model(model, X_val, y_val, label_encoder)
        print_evaluation(name, validation_results[name])

    best_model_name = max(
        validation_results.items(),
        key=lambda item: item[1]["weighted_f1"],
    )[0]
    best_model = models[best_model_name]

    print(f"\nBest model on validation set: {best_model_name}")

    print("\nFinal test evaluation")
    test_metrics = evaluate_model(best_model, X_test, y_test, label_encoder)
    print_evaluation(f"{best_model_name} (test)", test_metrics)

    save_artifacts(count_vec, tfidf_vec, best_model, label_encoder)


if __name__ == "__main__":
    main()
