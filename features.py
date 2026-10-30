import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly")
SUSPICIOUS_WORDS = (
    "verify", "urgent", "suspended", "password", "login", "account",
    "confirm", "update", "winner", "prize", "gift card", "bank details",
    "wire transfer", "click here", "limited time", "unusual activity"
)

def extract_url_features(email):
    text = str(email or "")
    urls = URL_RE.findall(text)
    lower = text.lower()

    url_count = len(urls)
    url_chars = sum(len(u) for u in urls)
    https_count = sum(u.lower().startswith("https://") for u in urls)
    ip_host_count = sum(bool(re.search(r"https?://(?:\d{1,3}\.){3}\d{1,3}", u, re.I)) for u in urls)
    shortener_count = sum(any(domain in u.lower() for domain in SHORTENERS) for u in urls)
    at_count = sum("@" in u for u in urls)
    hyphen_count = sum(u.count("-") for u in urls)
    suspicious_count = sum(word in lower for word in SUSPICIOUS_WORDS)
    return [
        url_count,
        url_chars,
        https_count,
        ip_host_count,
        shortener_count,
        at_count,
        hyphen_count,
        suspicious_count,
        int("!" in text),
        int(bool(re.search(r"\b(password|passcode|otp|one[- ]time code)\b", lower))),
    ]

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert email strings into simple numeric URL and wording indicators."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([extract_url_features(x) for x in X], dtype=float)

def build_feature_union():
    return FeatureUnion([
        ("text", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
            sublinear_tf=True
        )),
        ("url_features", URLFeatureExtractor()),
    ])
