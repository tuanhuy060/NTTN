import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =========================
# LOAD DATA
# =========================
print("START")
df = pd.read_csv("../train_with_aug_raw.csv")
print(df.head())
print(df.columns.tolist())

print("Dataset shape:", df.shape)

# =========================
# CHECK COLUMN
# =========================

print(df.columns)

TEXT_COL = "review"
LABEL_COL = "label"

# =========================
# REMOVE NULL
# =========================

df = df.dropna(subset=[TEXT_COL, LABEL_COL])

# =========================
# LABEL ENCODE
# =========================

label_map = {
    "negative": 0,
    "positive": 1
}

df[LABEL_COL] = df[LABEL_COL].map(label_map)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL],
    df[LABEL_COL],
    test_size=0.2,
    random_state=42,
    stratify=df[LABEL_COL]
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# =========================
# MODEL
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# =========================
# PREDICT
# =========================

preds = model.predict(X_test_tfidf)

# =========================
# METRICS
# =========================

acc = accuracy_score(y_test, preds)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "../models/logistic_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("\nSaved model successfully.")