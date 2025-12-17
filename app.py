from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

model = joblib.load("model/sentiment_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    if request.method == "POST":
        review = request.form["review"]
        review = clean_text(review)
        vector = tfidf.transform([review])
        sentiment = model.predict(vector)[0]

    return f"""
    <h2>Amazon Review Sentiment Analysis</h2>
    <form method="post">
        <textarea name="review" rows="5" cols="50"></textarea><br><br>
        <input type="submit">
    </form>
    <h3>Sentiment: {sentiment}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)
