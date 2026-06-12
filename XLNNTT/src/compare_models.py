import joblib

# LOAD MODELS
logistic_model = joblib.load("../models/logistic_model.pkl")
svm_model = joblib.load("../models/svm_model.pkl")

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

label_map = {
    0: "negative",
    1: "positive"
}

print("=== MODEL COMPARISON DEMO ===")

while True:
    text = input("\nNhập review: ")

    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])

    logistic_pred = logistic_model.predict(vec)[0]
    svm_pred = svm_model.predict(vec)[0]

    print("\nLogistic Regression:")
    print(label_map[logistic_pred])

    print("\nLinearSVC:")
    print(label_map[svm_pred])