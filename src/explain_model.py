import joblib
import pandas as pd

# ====================
# LOAD MODEL
# ====================

model = joblib.load("../models/svm_model.pkl")

# đổi tên nếu muốn xem Logistic
# model = joblib.load("../models/logistic_model.pkl")

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# ====================
# GET WORDS + WEIGHTS
# ====================

words = vectorizer.get_feature_names_out()

weights = model.coef_[0]

df = pd.DataFrame({
    "word": words,
    "score": weights
})

# ====================
# TOP POSITIVE
# ====================

print("\n===== TOP 20 POSITIVE WORDS =====")

print(
    df
    .sort_values(
        "score",
        ascending=False
    )
    .head(20)
)

# ====================
# TOP NEGATIVE
# ====================

print("\n===== TOP 20 NEGATIVE WORDS =====")

print(
    df
    .sort_values(
        "score"
    )
    .head(20)
)