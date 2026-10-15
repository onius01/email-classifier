"""
Train the spam classifier.

Usage:
    1. Put spam.csv (SMS Spam Collection, columns v1/v2) in ./data
    2. python train.py

Outputs (in ./model): spam_model.joblib, metrics.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from spam_utils import build_pipeline

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data" / "spam.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MIN_PRECISION = 0.98   # we'd rather miss some spam than flag a real message


def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")   # the dataset is NOT utf-8
    for label_col, text_col in [("v1", "v2"), ("Category", "Message"), ("label", "text")]:
        if label_col in df.columns and text_col in df.columns:
            df = df[[label_col, text_col]]
            df.columns = ["label", "text"]
            break
    else:
        sys.exit(f"Could not find label/text columns. Found: {list(df.columns)}")

    df = df.dropna().drop_duplicates()
    df["target"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    return df.reset_index(drop=True)


def pick_threshold(y_true, proba):
    """
    Among thresholds that keep precision >= MIN_PRECISION, find the best recall,
    then take the HIGHEST threshold that is within 0.5% of that recall
    (recall often plateaus; the higher threshold means fewer false alarms).
    Clipped to a sane 0.2-0.8 range.
    """
    valid = []
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        if precision_score(y_true, pred) >= MIN_PRECISION:
            valid.append((float(round(t, 2)), recall_score(y_true, pred)))
    if not valid:
        return 0.5
    best_recall = max(r for _, r in valid)
    t = max(t for t, r in valid if r >= best_recall - 0.005)
    return float(min(max(t, 0.2), 0.8))


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} unique messages, {df['target'].mean():.1%} spam")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2, stratify=df["target"], random_state=42
    )

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, class_weight="balanced", max_iter=2000),
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1, class_weight="balanced"), cv=3),
    }

    # 1) compare models with 5-fold cross-validation on the TRAINING set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, clf in candidates.items():
        scores = cross_validate(build_pipeline(clf), X_train, y_train, cv=cv,
                                scoring=["precision", "recall", "f1"])
        results[name] = {m: float(scores[f"test_{m}"].mean()) for m in ["precision", "recall", "f1"]}
        print(f"{name:26s} " + "  ".join(f"{m}={v:.4f}" for m, v in results[name].items()))

    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest by F1: {best_name}")

    # 2) tune the decision threshold using out-of-fold probabilities
    proba_cv = cross_val_predict(build_pipeline(candidates[best_name]), X_train, y_train,
                                 cv=cv, method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train.values, proba_cv)
    print(f"Chosen threshold: {threshold} (precision >= {MIN_PRECISION} on CV)")

    # 3) final fit + honest evaluation on the untouched test set
    pipe = build_pipeline(candidates[best_name]).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
        "n_messages": int(len(df)),
        "spam_ratio": float(df["target"].mean()),
        "model": best_name,
        "threshold": threshold,
        "cv_results": results,
        "test": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "confusion": {"true_ham": int(tn), "false_spam": int(fp),
                          "missed_spam": int(fn), "caught_spam": int(tp)},
        },
    }
    print("\nTest set:", json.dumps(metrics["test"], indent=2))

    joblib.dump(pipe, MODEL_DIR / "spam_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved model/spam_model.joblib and model/metrics.json")


if __name__ == "__main__":
    main()
