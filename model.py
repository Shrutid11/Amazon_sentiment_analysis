import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
stop_words = stopwords.words("english")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def train_model():
    df = pd.read_csv("data/amazon_reviews.csv", encoding="latin1")
    df.columns = df.columns.str.replace("ï»¿", "", regex=False)

    df["Sentiment"] = df["Positive"].apply(
        lambda x: "Positive" if x == 1 else "Negative"
    )

    df["cleaned_review"] = df["reviewText"].apply(clean_text)

    X = df["cleaned_review"]
    y = df["Sentiment"]

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "model/sentiment_model.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

    print("Model trained and saved successfully")

if __name__ == "__main__":
    train_model()
