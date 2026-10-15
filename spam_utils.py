"""
Shared code for training (train.py) and serving (app.py).

Everything the model needs - cleaning, features, classifier - lives in ONE
sklearn Pipeline, so the app can never preprocess text differently from training.
"""
import re
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

# ----------------------------------------------------------------- cleaning
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|info|biz|co\.uk)\b\S*)", re.I
)
MONEY_RE = re.compile(
    r"(?:[£$€₹]\s?\d[\d,.]*)|(?:\b\d[\d,.]*\s?(?:pounds?|gbp|usd|dollars?|rs\.?|inr|euros?)\b)",
    re.I,
)
LONG_NUM_RE = re.compile(r"\b\d{5,}\b")      # phone numbers, short codes
NUM_RE = re.compile(r"\d+")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@lru_cache(maxsize=None)
def _stem(word):
    """Tiny suffix stripper (running -> runn, claims -> claim). No NLTK needed."""
    if len(word) <= 4:
        return word
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def clean_text(text):
    """
    Lowercase, replace URLs / money / phone numbers with placeholder tokens
    (these are strong spam signals that the old isalnum() filter threw away),
    then tokenize and stem. Stop words are KEPT: words like 'to', 'now', 'you'
    are useful in phrases such as 'call now' or 'to claim'.
    """
    t = str(text).lower()
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = URL_RE.sub(" urltoken ", t)
    t = MONEY_RE.sub(" moneytoken ", t)
    t = LONG_NUM_RE.sub(" phonetoken ", t)
    t = NUM_RE.sub(" numtoken ", t)
    return " ".join(w if w.endswith("token") else _stem(w) for w in WORD_RE.findall(t))


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(t) for t in X]


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Hand-made signals: length, digits, CAPS ratio, '!' count, has URL / money / phone."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            t = str(t)
            n = max(len(t), 1)
            letters = max(sum(c.isalpha() for c in t), 1)
            rows.append([
                np.log1p(len(t)),
                np.log1p(len(t.split())),
                sum(c.isdigit() for c in t) / n,
                sum(c.isupper() for c in t) / letters,
                np.log1p(t.count("!")),
                float(bool(URL_RE.search(t))),
                float(bool(MONEY_RE.search(t))),
                float(bool(LONG_NUM_RE.search(t))),
            ])
        return np.array(rows, dtype=float)


# ----------------------------------------------------------------- pipeline
def build_pipeline(classifier):
    features = FeatureUnion([
        # words + word pairs ("call now", "claim prize")
        ("word", Pipeline([
            ("clean", TextCleaner()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ])),
        # character n-grams catch obfuscation like "fr33", "w1n", "cl@im"
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=3, sublinear_tf=True)),
        ("meta", Pipeline([("m", MetaFeatures()), ("scale", MaxAbsScaler())])),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


# ----------------------------------------------------------------- inference
def spam_probability(pipe, texts):
    return pipe.predict_proba(list(texts))[:, 1]


def verdict(prob, threshold):
    """Returns (label, level) where level is 'spam' | 'suspicious' | 'ham'."""
    if prob >= threshold:
        return "Likely spam", "spam"
    if prob >= threshold / 2:
        return "Suspicious - borderline", "suspicious"
    return "Looks legitimate", "ham"


def explain(pipe, text, k=6):
    """
    Top word-level signals behind the prediction: list of (term, contribution).
    Positive = pushes toward spam, negative = toward not-spam.
    Terms are shown in cleaned/stemmed form. Works for linear and Naive Bayes models.
    """
    union = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    word_pipe = dict(union.transformer_list)["word"]
    tfidf = word_pipe.named_steps["tfidf"]
    n_word = len(tfidf.vocabulary_)

    if hasattr(clf, "coef_"):
        w = clf.coef_[0][:n_word]
    elif hasattr(clf, "feature_log_prob_"):
        w = (clf.feature_log_prob_[1] - clf.feature_log_prob_[0])[:n_word]
    else:
        return []

    x = word_pipe.transform([text]).toarray().ravel()
    contrib = x * w
    names = tfidf.get_feature_names_out()
    idx = np.argsort(-np.abs(contrib))[:k]
    return [(names[i], float(contrib[i])) for i in idx if contrib[i] != 0]
