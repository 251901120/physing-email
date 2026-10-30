# Phishing Email Detection Model

A beginner-friendly Scikit-learn project that classifies email text as **Phishing** or **Safe** using TF-IDF text features, URL indicators, and Logistic Regression.

## 1. Project files

- `data/emails.csv` — starter dataset; replace or expand it with a properly labelled dataset for meaningful evaluation.
- `features.py` — text and URL feature extraction.
- `train_model.py` — trains and evaluates the classifier; saves the model, metrics, and confusion-matrix image.
- `app.py` — Streamlit interface for testing emails.
- `requirements.txt` — Python packages.

## 2. Setup in VS Code (Windows)

Open this project folder in VS Code. Open **Terminal → New Terminal**, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Dataset format

`data/emails.csv` must have:
- an email text column named `email_text` (also accepts `text`, `body`, `message`, or `content`)
- a label column named `label` (also accepts `class`, `category`, or `type`)

Labels can be `Phishing` / `Safe` (also accepts common equivalents such as `spam` / `ham` or `1` / `0`). Include both classes and enough examples of each. Keep the original dataset source and licence with your submission.

The included CSV is a small demonstration dataset to confirm that the code runs. It is **not** large enough to establish high real-world accuracy. For your final evaluation, replace/expand it with a reputable, labelled email dataset and record its source.

## 4. Train the model

In the VS Code terminal:

```powershell
python train_model.py
```

This creates:
- `phishing_model.joblib`
- `metrics.json`
- `confusion_matrix.png`

The terminal prints accuracy, precision, recall, and F1-score. Report the actual values produced by your own dataset; do not copy example values.

## 5. Run the web application

```powershell
streamlit run app.py
```

Streamlit will show a local URL (normally `http://localhost:8501`). Open it in your browser, paste an email, and click **Analyze Email**.

## 6. What the model does

1. TF-IDF converts email words and two-word phrases into numerical features.
2. URL features count links and identify indicators such as IP-address URLs, URL shorteners, `@` characters, and suspicious wording.
3. Logistic Regression learns patterns from labelled examples.
4. A held-out test split measures performance on examples not used for training.
5. A confusion matrix shows correct and incorrect predictions for both classes.

## 7. Submission checklist

- [ ] `data/emails.csv` and dataset source/licence
- [ ] `features.py`, `train_model.py`, and `app.py`
- [ ] `requirements.txt`
- [ ] `phishing_model.joblib`
- [ ] `confusion_matrix.png`
- [ ] Screenshot of Streamlit prediction for a phishing example
- [ ] Screenshot of Streamlit prediction for a legitimate example
- [ ] Screenshot of terminal accuracy and classification report
- [ ] Short report explaining dataset, features, algorithm, results, and limitations

**Security note:** A prediction of Safe does not prove an email is legitimate. Do not click suspicious links or share credentials based only on this prototype.
