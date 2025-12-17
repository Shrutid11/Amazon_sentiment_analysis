# Amazon Review Sentiment Analysis

## 📌 Project Overview
This project performs sentiment analysis on Amazon product reviews using
Natural Language Processing and Machine Learning.

The system classifies customer reviews into **Positive** or **Negative**
sentiments and provides real-time predictions via a Streamlit dashboard.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- NLP (NLTK)
- TF-IDF Vectorizer
- Logistic Regression
- Scikit-learn
- Streamlit

---

## 📂 Project Structure
Amazon_sentiment_analysis/
│
├── data/
├── model/
├── app.py
├── dashboard.py
├── model.py
├── sentiment_analysis.ipynb
├── requirements.txt
└── README.md


---

## ⚙️ How to Run the Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/Amazon_sentiment_analysis.git
cd Amazon_sentiment_analysis

2️⃣ Install Dependencies

python -m pip install -r requirements.txt

3️⃣ Train Model
python model.py

4️⃣ Run Streamlit App
python -m streamlit run dashboard.py

📊 Model Details

Feature Extraction: TF-IDF

Algorithm: Logistic Regression

Accuracy: ~89%


🧠 Key Learnings

Real-world dataset preprocessing

Handling encoding & schema issues

Text feature engineering using TF-IDF

Model serialization with joblib

Deployment using Streamlit

👩‍💻 Author

Shruti Dhote
Data Science Student