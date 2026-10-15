 HEAD
# Spam Classifier

SMS / short-email spam detector with a Streamlit UI.

## Run it
```bash
pip install -r requirements.txt
```
1. Put `spam.csv` (the SMS Spam Collection, columns `v1`/`v2`) in the `data/` folder.
2. Train: `python train.py`  (prints model comparison and test results, saves `model/`)
3. Launch: `streamlit run app.py`

## What's different from the notebook version
- One sklearn `Pipeline` holds cleaning + features + model, so training and the app can't drift apart.
- No NLTK downloads needed (works on any host without `punkt`/`stopwords` setup).
- Links, money amounts and phone numbers become tokens instead of being deleted.
- Word + character n-gram TF-IDF plus hand-made signals (CAPS, `!`, length...).
- Cross-validated model selection and a tuned threshold that favours precision.
- The vectorizer is fit only on training data (no leakage into the test set).
- App shows probability, an explanation of why, batch CSV scoring, and live model metrics.

## Limits
The dataset is ~5.5k English SMS from around 2012. It will not reflect modern phishing or
long HTML emails well. For better coverage, add your own labelled messages to the CSV and retrain.

## Deploying (Streamlit Community Cloud)
Commit `model/` (it is small). Set the Python version / `scikit-learn` pin to match the version
printed in `model/metrics.json` so the model loads cleanly.

# email-classifier
 224e7504328e347c91f5d37ccaf25eeb783bd97e
