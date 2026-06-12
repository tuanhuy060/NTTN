import joblib

model = joblib.load("../models/logistic_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

label_map = {
    0: "negative",
    1: "positive"
}

print("=== SENTIMENT ANALYZER ===")

while True:
    text = input("\nNhập review: ")

    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    print("Prediction:", label_map[pred])