import json
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from spam_utils import explain, spam_probability, verdict

st.set_page_config(page_title="Spam Classifier", page_icon="🛡️", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
MAX_BATCH_ROWS = 20000

EXAMPLES = {
    "(choose an example)": "",
    "Prize scam": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "Fake link": "Congratulations! You've won a free gift voucher. Click http://bit.ly/claim-now to get yours today!",
    "Normal chat": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "Tricky ham": "Call me when you get a chance, I won a small bet today lol. Nothing urgent.",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


if not (MODEL_PATH.exists() and METRICS_PATH.exists()):
    st.error("Model files not found. Run `python train.py` first (see README).")
    st.stop()

pipe, metrics = load_model()

if metrics.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {metrics['sklearn_version']} but you have "
        f"{sklearn.__version__}. If you see odd errors, re-run `python train.py`."
    )

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Spam threshold", 0.05, 0.95, float(metrics["threshold"]), 0.05,
        help="Messages scoring above this are called spam. Lower = catches more spam "
             "but flags more real messages. Default was tuned for high precision.",
    )
    st.divider()
    t = metrics["test"]
    st.subheader("Model quality")
    st.caption(f"{metrics['model']} - held-out test set")
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{t['precision']:.1%}")
    c2.metric("Recall", f"{t['recall']:.1%}")
    st.caption(
        f"Trained on {metrics['n_messages']:,} messages "
        f"({metrics['spam_ratio']:.0%} spam) on {metrics['trained_at']}."
    )

# --------------------------------------------------------------------- main
st.title("🛡️ Spam Classifier")
st.caption("Trained on short SMS messages. Very long emails or HTML newsletters are "
           "outside what it learned, so treat those results with extra caution.")

tab_single, tab_batch, tab_about = st.tabs(["Check a message", "Check a CSV file", "About"])

# ---- single message ----
with tab_single:
    example = st.selectbox("Try an example", list(EXAMPLES))
    text = st.text_area("Message", value=EXAMPLES[example], height=140,
                        placeholder="Paste an SMS or short email here...")

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter a message first.")
        else:
            prob = float(spam_probability(pipe, [text])[0])
            label, level = verdict(prob, threshold)

            if level == "spam":
                st.error(f"🚨 {label}")
            elif level == "suspicious":
                st.warning(f"⚠️ {label}")
            else:
                st.success(f"✅ {label}")

            st.metric("Spam probability", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))

            if len(text) > 1000:
                st.info("This message is much longer than the SMS the model was trained on.")

            signals = explain(pipe, text, k=6)
            if signals:
                st.subheader("Why?")
                st.caption("Strongest word signals (shown in cleaned form: "
                           "`moneytoken` = an amount of money, `phonetoken` = a long number, "
                           "`urltoken` = a link).")
                st.dataframe(
                    pd.DataFrame({
                        "Signal": [s for s, _ in signals],
                        "Pushes toward": ["Spam" if c > 0 else "Not spam" for _, c in signals],
                        "Strength": [round(abs(c), 3) for _, c in signals],
                    }),
                    hide_index=True,
                )

# ---- batch ----
with tab_batch:
    st.write("Upload a CSV with a column of messages and get a spam score for every row.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = up.getvalue()
        try:
            df = pd.read_csv(StringIO(raw.decode("utf-8")))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(raw.decode("latin-1")))

        if df.empty:
            st.warning("That file has no rows.")
        else:
            default = next((i for i, c in enumerate(df.columns)
                            if str(c).lower() in ("text", "message", "v2", "body", "sms")), 0)
            col = st.selectbox("Which column holds the message?", df.columns, index=default)
            if len(df) > MAX_BATCH_ROWS:
                st.info(f"Only the first {MAX_BATCH_ROWS:,} rows will be scored.")
                df = df.head(MAX_BATCH_ROWS)

            if st.button("Score all messages"):
                texts = df[col].fillna("").astype(str)
                probs = spam_probability(pipe, texts)
                out = df.copy()
                out["spam_probability"] = probs.round(4)
                out["prediction"] = ["spam" if p >= threshold else "ham" for p in probs]

                n_spam = int((out["prediction"] == "spam").sum())
                st.success(f"Scored {len(out):,} messages: {n_spam:,} flagged as spam "
                           f"({n_spam / len(out):.1%}).")
                st.dataframe(out.sort_values("spam_probability", ascending=False))
                st.download_button("Download results (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "spam_results.csv", "text/csv")

# ---- about ----
with tab_about:
    st.markdown(
        """
**How it works**

1. **Clean:** lowercase, and replace links, money amounts and phone numbers with
   special tokens (they are strong spam signals), then tokenize and stem.
2. **Features:** word and word-pair TF-IDF, character n-grams (catches tricks like
   `fr33` or `w1n`), plus signals such as length, CAPS ratio, `!` count.
3. **Model:** three classifiers were compared with 5-fold cross-validation and the best
   was kept. The decision threshold is tuned to keep precision high, since flagging a
   real message as spam is worse than missing a spam message.
        """
    )
    c = metrics["test"]["confusion"]
    st.markdown("**Test-set results**")
    st.dataframe(
        pd.DataFrame(
            {"Predicted not spam": [c["true_ham"], c["missed_spam"]],
             "Predicted spam": [c["false_spam"], c["caught_spam"]]},
            index=["Actually not spam", "Actually spam"],
        )
    )
    st.markdown("**Cross-validation (training set)**")
    st.dataframe(pd.DataFrame(metrics["cv_results"]).T.round(4))
