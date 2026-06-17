from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load models
svm = joblib.load("../models/svm_model.pkl")
logistic = joblib.load("../models/logistic_model.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")


# =========================
# WEB UI
# =========================
@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# =========================
# COMPARE API
# =========================
@app.route("/api/compare", methods=["POST"])
def compare():

    data = request.get_json(force=True)

    text = data["text"]

    vec = tfidf.transform([text])

    p1 = logistic.predict(vec)[0]
    p2 = svm.predict(vec)[0]

    logistic_result = (
        "Positive"
        if p1 == 1
        else "Negative"
    )

    svm_result = (
        "Positive"
        if p2 == 1
        else "Negative"
    )

    return jsonify({

        "input": text,

        "logistic":
        logistic_result,

        "svm":
        svm_result

    })


# =========================
# API STATUS
# =========================
@app.route("/api/predict", methods=["GET"])
def api_status():

    return jsonify({

        "message":
        "API Running"

    })


# =========================
# RUN
# =========================
if __name__ == "__main__":

    app.run(
        debug=True
    )