# Nepali Political Sentiment Analysis

A Streamlit-based web application that analyzes sentiment in Nepali-language political tweets using classical Machine Learning. No Twitter API token required — tweets are scraped freely via Nitter instances.

---

## 📌 Overview

This project performs **binary sentiment classification** (Positive / Negative) on Nepali-script political tweets. It combines a trained sklearn ML model with a live tweet scraper and an interactive dashboard to visualize sentiment trends in Nepali political discourse.

---

## ✨ Features

- 🔍 **Live Tweet Scraping** — Scrapes Nepali tweets in real time via multiple Nitter instances (no Twitter API needed)
- 🧠 **Classical ML Inference** — Uses a trained sklearn classifier with TF-IDF + CountVectorizer features
- 📝 **Paste Text Mode** — Analyze any custom Nepali text directly without scraping
- 📊 **Interactive Dashboard** — Pie charts, confidence histograms, and sentiment metric cards powered by Plotly
- ☁️ **Word Clouds** — Visual word frequency breakdowns for Positive, Negative, and All tweets
- 🌐 **Nepali-script Filter** — Only retains tweets with ≥25% Devanagari characters
- ⬇️ **CSV Export** — Download full results with sentiment labels and confidence scores

---

## 🗂️ Project Structure

```
Sentiment_Analysis_Politics/
│
├── app.py                          # Main Streamlit application
├── train_tfidf_classifiers.py      # Model training script
├── wordcloud_component.py          # Word cloud rendering component
├── datacleaning.ipynb              # Data cleaning notebook
├── nepali_sentiment_classical_ml.ipynb  # ML model training notebook
│
├── best_ml_model/                  # Saved model artifacts
│   ├── classifier.pkl              # Trained sklearn classifier
│   ├── count_vectorizer.pkl        # CountVectorizer
│   ├── tfidf_vectorizer.pkl        # TF-IDF Vectorizer
│   └── label_map.json              # Label mapping (0=Negative, 1=Positive)
│
├── Label/                          # Labeled training data
├── cleaned_dataset.csv             # Preprocessed dataset
├── unique_tweets.xlsx              # Raw collected tweets
├── stop_words_nepali.txt           # Nepali stopwords list
└── style.css                       # Custom UI styling
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Web App** | Streamlit |
| **ML / NLP** | scikit-learn, scipy, TF-IDF, CountVectorizer |
| **Scraping** | Selenium, webdriver-manager, Nitter |
| **Visualization** | Plotly, WordCloud, Matplotlib |
| **Data** | Pandas, NumPy |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/leninLD/Sentiment_Analysis_Politics.git
cd Sentiment_Analysis_Politics
```

### 2. Install dependencies

```bash
pip install streamlit selenium webdriver-manager scikit-learn scipy wordcloud matplotlib pandas numpy plotly pillow unicodedata2
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

### Data Pipeline
1. **Data Collection** — Tweets collected and stored in `unique_tweets.xlsx`
2. **Data Cleaning** (`datacleaning.ipynb`) — Removes URLs, mentions, hashtags, emojis, noise characters, and zero-width characters; retains only Devanagari text
3. **Model Training** (`nepali_sentiment_classical_ml.ipynb`) — Trains classical ML classifiers using combined Count + TF-IDF features on the cleaned dataset

### Inference
- Input text is cleaned using the same pipeline as training
- Features are extracted using the saved `count_vectorizer.pkl` and `tfidf_vectorizer.pkl`
- The classifier predicts **Positive** or **Negative** with a confidence score via `predict_proba`

### App Tabs
| Tab | Description |
|---|---|
| 🏠 Home | Overview dashboard |
| 🔍 Search Twitter | Scrape live Nepali tweets by keyword and analyze |
| 📝 Paste Text | Paste any Nepali text lines for instant analysis |

---

## 📊 Model

- **Feature Extraction:** CountVectorizer + TF-IDF Vectorizer (concatenated features)
- **Tokenizer:** Whitespace tokenizer tuned for Devanagari script
- **Labels:** `0 = Negative`, `1 = Positive`
- **Saved artifacts:** `best_ml_model/` folder (joblib format)

---

## ⚙️ Configuration

In the sidebar of the app you can configure:
- **Model folder path** — Path to your `best_ml_model` folder
- **Max tweets to fetch** — Slider from 10 to 150 tweets

---

## 📋 Requirements

- Python 3.10+
- Google Chrome (for Selenium-based scraping)
- ChromeDriver (auto-managed via `webdriver-manager`)

---

## 👤 Author

**Lenin D L**
- GitHub: [@leninLD](https://github.com/leninLD)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).