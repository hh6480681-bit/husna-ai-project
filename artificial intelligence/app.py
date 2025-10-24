from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle

app = Flask(__name__)

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Hardcoded demo login
DEMO_USER = "user"
DEMO_PASS = "1234"

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == DEMO_USER and password == DEMO_PASS:
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error="")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get text from JSON (fetch API)
    if request.is_json:
        data = request.get_json()
        text = (data.get("text") or "").strip()
    else:
        text = (request.form.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No input provided"}), 400

    # Transform text and predict
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    # Map numeric label to human-readable
    label = "Real" if pred == 1 else "Fake"

    return jsonify({"prediction": int(pred), "label": label})

if __name__ == "__main__":
    app.run(debug=True)
