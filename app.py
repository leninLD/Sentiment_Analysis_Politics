"""
🇳🇵 Nepali Tweet Sentiment Analyzer
=====================================
No Twitter API token needed — scrapes via Nitter instances (free).

Install:
    pip install streamlit selenium webdriver-manager scikit-learn scipy
                wordcloud matplotlib pandas numpy plotly pillow unicodedata2

Run:
    streamlit run app.py
"""

import os
import re
import time
import random
import unicodedata
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
from wordcloud_component import make_wordcloud_html

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

NEPALI_FONT_PATH = "fonts/NotoSansDevanagari-VariableFont.ttf"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nepali Tweet Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD EXTERNAL CSS
# ─────────────────────────────────────────────────────────────────────────────
def load_css(path: str = "style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ CSS file not found at `{path}`. UI may look unstyled.")

load_css("style.css")

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR  (rendered at the very top, before any other content)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# NITTER INSTANCES
# ─────────────────────────────────────────────────────────────────────────────
NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.lucahammer.com",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
    "https://nitter.net",
    "https://nitter.nl",
    "https://nitter.unixfox.eu",
]

# Devanagari range for Nepali-language filtering
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def _is_nepali(text: str, min_ratio: float = 0.25) -> bool:
    if not text:
        return False
    dev_chars    = len(_DEVANAGARI_RE.findall(text))
    total_letters = sum(1 for ch in text if unicodedata.category(ch).startswith('L'))
    if total_letters == 0:
        return False
    return (dev_chars / total_letters) >= min_ratio


# ─────────────────────────────────────────────────────────────────────────────
# SELENIUM DRIVER SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )
    driver.execute_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    )
    return driver


# ─────────────────────────────────────────────────────────────────────────────
# FIND WORKING NITTER INSTANCE
# ─────────────────────────────────────────────────────────────────────────────
def get_working_instance(driver) -> str | None:
    from selenium.webdriver.common.by import By
    for instance in NITTER_INSTANCES:
        try:
            driver.get(instance)
            time.sleep(3)
            driver.find_element(By.CSS_SELECTOR, 'input[name="q"]')
            return instance
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NITTER SCRAPER — Nepali-only filter applied here
# ─────────────────────────────────────────────────────────────────────────────
def scrape_nitter(keyword: str, target: int = 50, status_box=None) -> list[dict]:
    from selenium.webdriver.common.by import By

    def log(msg):
        if status_box:
            status_box.info(msg)

    driver    = setup_driver()
    collected = []
    seen_text = set()

    try:
        log("🔍 Finding a live Nitter instance…")
        base_url = get_working_instance(driver)
        if not base_url:
            log("❌ All Nitter instances are down. Try again in a few minutes.")
            return []

        log(f"✅ Connected to: {base_url}")
        # Append lang:ne to hint Nitter; _is_nepali() acts as a hard filter below
        encoded     = (keyword + " lang:ne").replace(" ", "+").replace('"', "")
        next_cursor = None
        page        = 1
        skipped     = 0

        while len(collected) < target:
            url = (
                f"{base_url}/search?f=tweets&q={encoded}&cursor={next_cursor}"
                if next_cursor
                else f"{base_url}/search?f=tweets&q={encoded}"
            )
            driver.get(url)
            time.sleep(random.uniform(3, 5))

            articles = driver.find_elements(By.CSS_SELECTOR, ".timeline-item")
            if not articles:
                articles = driver.find_elements(By.CSS_SELECTOR, ".tweet-body")

            log(
                f"📄 Page {page} — {len(articles)} items | "
                f"{len(collected)}/{target} Nepali tweets collected "
                f"({skipped} non-Nepali skipped)…"
            )

            new_this_page = 0
            for article in articles:
                try:
                    try:
                        text = article.find_element(
                            By.CSS_SELECTOR, ".tweet-content"
                        ).text.strip()
                    except Exception:
                        text = article.text.strip()

                    if len(text) < 10:
                        continue
                    clean = text.replace("\n", " ").strip()
                    if clean in seen_text:
                        continue

                    # Hard Nepali-language filter
                    if not _is_nepali(clean):
                        skipped += 1
                        continue

                    seen_text.add(clean)

                    try:
                        path = article.find_element(
                            By.CSS_SELECTOR, ".tweet-date a"
                        ).get_attribute("href")
                        tweet_url = (
                            path if path.startswith("http") else base_url + path
                        )
                    except Exception:
                        tweet_url = ""

                    collected.append({"text": clean, "url": tweet_url})
                    new_this_page += 1

                except Exception:
                    continue

            if len(collected) >= target:
                break
            if new_this_page == 0:
                log("ℹ️ No new tweets on this page. Done.")
                break

            try:
                next_btn    = driver.find_element(By.CSS_SELECTOR, ".show-more a")
                href        = next_btn.get_attribute("href")
                next_cursor = (
                    href.split("cursor=")[-1] if "cursor=" in href else None
                )
                if not next_cursor:
                    break
            except Exception:
                break

            page += 1
            time.sleep(random.uniform(2, 4))

    except Exception as e:
        log(f"❌ Scraper error: {e}")
    finally:
        driver.quit()

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
_NOISE = (
    r'~؟॥"▬`%´•●=+÷।–][{}*"_…'
    + r"''\/"
    + ';abcdefghijklmnopqrstuvwxyz1234567890०१२३४५६७८९()-.|!?\",:—?।'
    + "'"
)

def _rm_emojis(t: str) -> str:
    cleaned = []
    for char in t:
        cp = ord(char)
        if (
            0x1F600 <= cp <= 0x1F64F or 0x1F300 <= cp <= 0x1F5FF or
            0x1F680 <= cp <= 0x1F6FF or 0x1F700 <= cp <= 0x1F77F or
            0x1F780 <= cp <= 0x1F7FF or 0x1F800 <= cp <= 0x1F8FF or
            0x1F900 <= cp <= 0x1F9FF or 0x1FA00 <= cp <= 0x1FA6F or
            0x1FA70 <= cp <= 0x1FAFF or 0x2600  <= cp <= 0x26FF  or
            0x2700  <= cp <= 0x27BF  or 0x2300  <= cp <= 0x23FF  or
            0x2B50  <= cp <= 0x2B55  or 0x25AA  <= cp <= 0x25FE  or
            0x1F1E0 <= cp <= 0x1F1FF or 0x1F100 <= cp <= 0x1F1FF or
            0x1F200 <= cp <= 0x1F2FF or 0x1F000 <= cp <= 0x1F02F or
            0x1F0A0 <= cp <= 0x1F0FF or 0xFE00  <= cp <= 0xFE0F  or
            0x1F3FB <= cp <= 0x1F3FF or 0x2702  <= cp <= 0x27B0  or
            cp in (0x200D, 0xFE0F, 0x00A9, 0x00AE, 0x2122,
                   0x3030, 0x303D, 0x3297, 0x3299) or
            unicodedata.category(char) in ('So', 'Sk')
        ):
            continue
        cleaned.append(char)
    return "".join(cleaned)

def _rm_zero_width(t: str) -> str:
    return re.sub(r'[\u200c-\u200f\u202a-\u202f\u2066-\u2069]', '', t)

def _rm_urls(t: str) -> str:
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'www\.\S+', ' ', t)
    t = re.sub(r't\.co/\S+', ' ', t)
    t = re.sub(r'\b\S+\.co\.np\S*', ' ', t)
    t = re.sub(r'\b\S+\.(com|org|net|io|gov|edu|co|info|biz|np|me|tv)\S*',
               ' ', t, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', t).strip()

def _rm_mentions(t: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'@\S*', '', t)).strip()

def _rm_hashtags(t: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'#\S*', '', t)).strip()

def _rm_noise_chars(t: str) -> str:
    t = t.lower()
    for ch in _NOISE:
        t = t.replace(ch, ' ')
    return re.sub(r'\s+', ' ', t).strip()

def _rm_special_chars(t: str) -> str:
    t = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def clean_tweet(text: str) -> str:
    if not text or str(text).strip() in ('', 'nan'):
        return ''
    t = str(text)
    t = _rm_emojis(t)
    t = _rm_zero_width(t)
    t = _rm_urls(t)
    t = _rm_mentions(t)
    t = _rm_hashtags(t)
    t = _rm_noise_chars(t)
    t = _rm_special_chars(t)
    return t.strip()


# ─────────────────────────────────────────────────────────────────────────────
# NEPALI TOKENIZER  — must match the function used during training exactly,
# so joblib can deserialize the TF-IDF vectorizer that references it.
# ─────────────────────────────────────────────────────────────────────────────
def nepali_tokenizer(text):
    return text.split(' ')


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  — Traditional ML (joblib / sklearn)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Loads the traditional ML model artefacts from *model_path* folder.
    Expected files (saved with joblib.dump):
        classifier.pkl        — trained sklearn classifier
        count_vectorizer.pkl  — CountVectorizer
        tfidf_vectorizer.pkl  — TfidfVectorizer
        label_map.json        — {"0": "Positive", "1": "Negative"}
    Returns (classifier, count_vec, tfidf_vec, label_map_dict).
    """
    import joblib, json

    def _load(fname):
        return joblib.load(os.path.join(model_path, fname))

    clf       = _load("classifier.pkl")
    count_vec = _load("count_vectorizer.pkl")
    tfidf_vec = _load("tfidf_vectorizer.pkl")

    label_map_path = os.path.join(model_path, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)   # {"0": "Positive", "1": "Negative"}

    return clf, count_vec, tfidf_vec, label_map


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT INFERENCE  — Traditional ML
# ─────────────────────────────────────────────────────────────────────────────
def predict_sentiment_batch(
    texts: list,
    clf,
    count_vec,
    tfidf_vec,
    label_map: dict,
    *_ignored,            # absorbs any extra positional args for compatibility
) -> list:
    """
    Predicts sentiment for a list of raw texts using the sklearn pipeline.

    The feature vector is the concatenation of count-vectorizer and
    tfidf-vectorizer outputs (matching typical training setups). If your
    training used only one of them the unused one simply contributes zeros,
    so the shapes will still match — adjust if needed.
    """
    from scipy.sparse import hstack
    import numpy as np

    results = []
    for text in texts:
        cleaned = clean_tweet(text)
        if not cleaned:
            results.append({
                "label": "POSITIVE", "confidence": 0.5,
                "pos_score": 0.5, "neg_score": 0.5,
            })
            continue

        count_feat = count_vec.transform([cleaned])
        tfidf_feat = tfidf_vec.transform([cleaned])
        features   = hstack([count_feat, tfidf_feat])

        pred_idx   = int(clf.predict(features)[0])
        label_str  = label_map.get(str(pred_idx), str(pred_idx)).upper()
        # label_map: {"0": "NEGATIVE", "1": "POSITIVE"}  →  0=NEG, 1=POS

        # Use predict_proba when available; fall back to a hard 1/0 score
        if hasattr(clf, "predict_proba"):
            probs     = clf.predict_proba(features)[0]   # shape (n_classes,)
            classes   = [int(c) for c in clf.classes_]
            # 0 → NEGATIVE, 1 → POSITIVE
            pos_idx_in_probs = classes.index(1) if 1 in classes else 1
            neg_idx_in_probs = classes.index(0) if 0 in classes else 0
            pos_score  = float(probs[pos_idx_in_probs]) if len(probs) > pos_idx_in_probs else 0.0
            neg_score  = float(probs[neg_idx_in_probs]) if len(probs) > neg_idx_in_probs else 0.0
            confidence = float(probs[classes.index(pred_idx)]) if pred_idx in classes else 0.5
        else:
            # Decision-function based confidence (SVM, etc.)
            confidence = 0.85
            pos_score  = 1.0 if label_str == "POSITIVE" else 0.0
            neg_score  = 0.0 if label_str == "POSITIVE" else 1.0

        results.append({
            "label"     : label_str,
            "confidence": confidence,
            "pos_score" : pos_score,
            "neg_score" : neg_score,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# NEPALI STOPWORDS
# ─────────────────────────────────────────────────────────────────────────────
def load_nepali_stopwords(path: str = "stop_words_nepali.txt") -> set:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()}
    except FileNotFoundError:
        return set()
    except Exception:
        return set()

NEPALI_STOPWORDS = load_nepali_stopwords("stop_words_nepali.txt")



# ─────────────────────────────────────────────────────────────────────────────
# TWEET CARD HTML  — pure sentiment, no emotion badges
# ─────────────────────────────────────────────────────────────────────────────
def tweet_card_html(row: dict, sentiment: str = "pos") -> str:
    label    = row["label"]          # "POSITIVE" | "NEGATIVE"
    conf     = row["confidence"]
    short    = row["text"][:240] + ("…" if len(row["text"]) > 240 else "")
    url      = row.get("url", "#") or "#"

    card_cls = "tweet-card-pos" if sentiment == "pos" else "tweet-card-neg"
    pill_cls = "pill-pos"        if sentiment == "pos" else "pill-neg"

    return f"""
    <div class="{card_cls}">
        <div class="tweet-text">{short}</div>
        <div class="tweet-footer">
            <span class="sentiment-pill {pill_cls}">
                {label}&nbsp; {conf*100:.0f}%
            </span>
            <a class="tweet-link" href="{url}" target="_blank">View Tweet ↗</a>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE PASTED TEXTS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pasted_texts(texts: list, clf, count_vec, tfidf_vec, label_map):
    sentiments = predict_sentiment_batch(texts, clf, count_vec, tfidf_vec, label_map)
    return pd.DataFrame({
        "text"      : texts,
        "url"       : [""] * len(texts),
        "label"     : [s["label"]       for s in sentiments],
        "confidence": [s["confidence"]  for s in sentiments],
        "pos_score" : [s["pos_score"]   for s in sentiments],
        "neg_score" : [s["neg_score"]   for s in sentiments],
    })


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY PASTED TEXT RESULTS  (Tab 2 — sentiment verdict + word clouds only)
# ─────────────────────────────────────────────────────────────────────────────
def display_pasted_results(df: pd.DataFrame):
    """
    Simplified view for pasted text analysis:
      1. Animated sentiment verdict banner (GIF-style CSS animation)
      2. Per-line sentiment breakdown cards
      3. Positive & Negative word clouds side-by-side
    """
    df["label"] = df["label"].str.upper()
    pos_df  = df[df["label"] == "POSITIVE"].sort_values("confidence", ascending=False)
    neg_df  = df[df["label"] == "NEGATIVE"].sort_values("confidence", ascending=False)
    total   = len(df)
    n_pos   = len(pos_df)
    n_neg   = len(neg_df)
    pct_pos = n_pos / total * 100 if total > 0 else 0
    pct_neg = n_neg / total * 100 if total > 0 else 0

    # ── Overall verdict (dominant sentiment) ─────────────────────────────
    if n_pos >= n_neg:
        verdict       = "POSITIVE"
        verdict_color = "#059669"
        verdict_bg    = "#ecfdf5"
        verdict_bdr   = "#6ee7b7"
        verdict_msg   = "The text carries an overall Positive sentiment 😊"
        conf_val      = pct_pos
        bar_color     = "#10b981"
        # Animated SVG: bouncing smiling face rings
        gif_html = """
        <svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
          <style>
            .ring1{animation:pulseRing 1.6s ease-in-out infinite;}
            .ring2{animation:pulseRing 1.6s ease-in-out infinite 0.3s;}
            .ring3{animation:pulseRing 1.6s ease-in-out infinite 0.6s;}
            .face {animation:bounceFace 1.6s ease-in-out infinite;}
            @keyframes pulseRing{0%,100%{opacity:.15;r:42;}50%{opacity:.35;r:48;}}
            @keyframes bounceFace{0%,100%{transform:translateY(0);}50%{transform:translateY(-6px);}}
          </style>
          <circle class="ring3" cx="60" cy="60" r="48" fill="none" stroke="#6ee7b7" stroke-width="2"/>
          <circle class="ring2" cx="60" cy="60" r="42" fill="none" stroke="#34d399" stroke-width="2"/>
          <circle class="ring1" cx="60" cy="60" r="36" fill="none" stroke="#10b981" stroke-width="2"/>
          <g class="face">
            <circle cx="60" cy="60" r="28" fill="#d1fae5" stroke="#10b981" stroke-width="2.5"/>
            <!-- eyes -->
            <circle cx="51" cy="55" r="3.5" fill="#059669"/>
            <circle cx="69" cy="55" r="3.5" fill="#059669"/>
            <!-- smile -->
            <path d="M48 66 Q60 78 72 66" fill="none" stroke="#059669" stroke-width="2.8"
                  stroke-linecap="round"/>
          </g>
        </svg>"""
    else:
        verdict       = "NEGATIVE"
        verdict_color = "#dc2626"
        verdict_bg    = "#fff5f5"
        verdict_bdr   = "#fca5a5"
        verdict_msg   = "The text carries an overall Negative sentiment 😔"
        conf_val      = pct_neg
        bar_color     = "#ef4444"
        # Animated SVG: shaking frowning face
        gif_html = """
        <svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
          <style>
            .ring1{animation:pulseRingR 1.6s ease-in-out infinite;}
            .ring2{animation:pulseRingR 1.6s ease-in-out infinite 0.3s;}
            .ring3{animation:pulseRingR 1.6s ease-in-out infinite 0.6s;}
            .face {animation:shakeFace 0.6s ease-in-out infinite;}
            @keyframes pulseRingR{0%,100%{opacity:.15;r:42;}50%{opacity:.35;r:48;}}
            @keyframes shakeFace{0%,100%{transform:translateX(0);}25%{transform:translateX(-5px);}75%{transform:translateX(5px);}}
          </style>
          <circle class="ring3" cx="60" cy="60" r="48" fill="none" stroke="#fca5a5" stroke-width="2"/>
          <circle class="ring2" cx="60" cy="60" r="42" fill="none" stroke="#f87171" stroke-width="2"/>
          <circle class="ring1" cx="60" cy="60" r="36" fill="none" stroke="#ef4444" stroke-width="2"/>
          <g class="face">
            <circle cx="60" cy="60" r="28" fill="#fee2e2" stroke="#ef4444" stroke-width="2.5"/>
            <!-- eyes -->
            <circle cx="51" cy="54" r="3.5" fill="#dc2626"/>
            <circle cx="69" cy="54" r="3.5" fill="#dc2626"/>
            <!-- frown -->
            <path d="M48 70 Q60 58 72 70" fill="none" stroke="#dc2626" stroke-width="2.8"
                  stroke-linecap="round"/>
          </g>
        </svg>"""

    # ── Render verdict banner ─────────────────────────────────────────────
    banner_html = f"""
    <div style="
        background:{verdict_bg};
        border:2px solid {verdict_bdr};
        border-radius:20px;
        padding:2rem 2.5rem;
        margin:1.5rem 0;
        display:flex;
        align-items:center;
        gap:2.5rem;
        box-shadow:0 4px 20px {verdict_color}18;
    ">
        <div style="flex-shrink:0;">
            {gif_html}
        </div>

        <div style="flex:1;">
            <div style="
                font-size:0.78rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:{verdict_color};margin-bottom:0.35rem;
            ">Overall Sentiment</div>
            <div style="
                font-size:2rem;font-weight:800;color:{verdict_color};
                letter-spacing:-0.5px;line-height:1.1;margin-bottom:0.4rem;
            ">{verdict}</div>
            <div style="font-size:0.95rem;color:#475569;margin-bottom:1rem;">{verdict_msg}</div>

            <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.3rem;font-weight:600;">
                Confidence: {conf_val:.0f}%
            </div>
            <div style="background:#e2e8f0;border-radius:999px;height:10px;overflow:hidden;width:100%;max-width:420px;">
                <div style="
                    height:100%;border-radius:999px;
                    background:linear-gradient(90deg,{bar_color},{verdict_bdr});
                    width:{conf_val:.0f}%;
                    animation:growBar 1.2s cubic-bezier(.4,0,.2,1) forwards;
                "></div>
            </div>

            <div style="display:flex;gap:1.5rem;margin-top:1rem;">
                <div style="
                    background:#fff;border:1px solid #bbf7d0;border-radius:10px;
                    padding:0.5rem 1rem;text-align:center;min-width:80px;
                ">
                    <div style="font-size:1.4rem;font-weight:700;color:#059669;">{n_pos}</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:600;">POSITIVE</div>
                </div>
                <div style="
                    background:#fff;border:1px solid #fecaca;border-radius:10px;
                    padding:0.5rem 1rem;text-align:center;min-width:80px;
                ">
                    <div style="font-size:1.4rem;font-weight:700;color:#dc2626;">{n_neg}</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:600;">NEGATIVE</div>
                </div>
                <div style="
                    background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:0.5rem 1rem;text-align:center;min-width:80px;
                ">
                    <div style="font-size:1.4rem;font-weight:700;color:#475569;">{total}</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:600;">TOTAL</div>
                </div>
            </div>
        </div>
    </div>

    <style>
    @keyframes growBar {{
        from {{ width: 0%; }}
        to   {{ width: {conf_val:.0f}%; }}
    }}
    </style>
    """
    
    # Apply the fix here to prevent Streamlit from treating indented HTML as Markdown code blocks
    st.markdown(banner_html.replace('\n', ''), unsafe_allow_html=True)

    for _, row in df.iterrows():
        lbl  = row["label"]
        conf = row["confidence"]
        text = row["text"][:300] + ("…" if len(row["text"]) > 300 else "")

        if lbl == "POSITIVE":
            card_cls  = "tweet-card-pos"
            pill_cls  = "pill-pos"
            icon      = "✅"
            bar_c     = "#10b981"
        else:
            card_cls  = "tweet-card-neg"
            pill_cls  = "pill-neg"
            icon      = "❌"
            bar_c     = "#ef4444"

        st.markdown(f"""
        <div class="{card_cls}" style="margin-bottom:0.7rem;">
            <div class="tweet-text">{text}</div>
            <div class="tweet-footer">
                <span class="sentiment-pill {pill_cls}">
                    {icon} {lbl} &nbsp; {conf*100:.0f}%
                </span>
                <div style="
                    background:#e2e8f0;border-radius:999px;height:6px;
                    overflow:hidden;width:180px;margin-left:auto;
                ">
                    <div style="
                        height:100%;border-radius:999px;
                        background:{bar_c};width:{conf*100:.0f}%;
                    "></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Word clouds (Positive & Negative only) ───────────────────────────
    st.markdown('<div class="section-title">☁️ Word Clouds</div>',
                unsafe_allow_html=True)

    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(
            '<div class="wc-panel-title" style="color:#059669;">🟢 Positive Words</div>',
            unsafe_allow_html=True
        )
        if n_pos > 0:
            make_wordcloud_html(pos_df["text"].tolist(),  "Greens", key="wc_pos")
        else:
            st.markdown(
                "<div style='background:#f0fdf4;border:1px solid #bbf7d0;"
                "border-radius:12px;padding:2rem;text-align:center;"
                "color:#64748b;font-size:0.9rem;'>No positive lines found</div>",
                unsafe_allow_html=True
            )
    with wc2:
        st.markdown(
            '<div class="wc-panel-title" style="color:#dc2626;">🔴 Negative Words</div>',
            unsafe_allow_html=True
        )
        if n_neg > 0:
            make_wordcloud_html(neg_df["text"].tolist(),  "Reds",   key="wc_neg")
        else:
            st.markdown(
                "<div style='background:#fff5f5;border:1px solid #fecaca;"
                "border-radius:12px;padding:2rem;text-align:center;"
                "color:#64748b;font-size:0.9rem;'>No negative lines found</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS  (shared by both tabs)
# ─────────────────────────────────────────────────────────────────────────────
def display_results(df, keyword="Pasted Texts"):
    # Normalise labels to uppercase so comparisons are robust
    df["label"] = df["label"].str.upper()

    pos_df  = df[df["label"] == "POSITIVE"].sort_values("confidence", ascending=False)
    neg_df  = df[df["label"] == "NEGATIVE"].sort_values("confidence", ascending=False)
    total   = len(df)
    n_pos   = len(pos_df)
    n_neg   = len(neg_df)
    pct_pos = n_pos / total * 100 if total > 0 else 0
    pct_neg = n_neg / total * 100 if total > 0 else 0

    # ── Metric cards ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""
        <div class="dash-card">
            <div class="icon-wrapper icon-blue">📊</div>
            <div>
                <div class="dash-value">{total}</div>
                <div class="dash-label">Total Tweets</div>
            </div>
        </div>""", unsafe_allow_html=True)
    m2.markdown(f"""
        <div class="dash-card">
            <div class="icon-wrapper icon-green">✅</div>
            <div>
                <div class="dash-value">{n_pos}</div>
                <div class="dash-label">Positive Tweets</div>
            </div>
            <div class="dash-badge badge-green">{pct_pos:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    m3.markdown(f"""
        <div class="dash-card">
            <div class="icon-wrapper icon-red">❌</div>
            <div>
                <div class="dash-value">{n_neg}</div>
                <div class="dash-label">Negative Tweets</div>
            </div>
            <div class="dash-badge badge-red">{pct_neg:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    m4.markdown(f"""
        <div class="dash-card">
            <div class="icon-wrapper icon-blue">🥧</div>
            <div>
                <div class="dash-value">{pct_pos:.0f}% / {pct_neg:.0f}%</div>
                <div class="dash-label">Positive / Negative</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Sentiment Distribution</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=["Negative", "Positive"],
            values=[pct_neg, pct_pos],
            hole=0.62,
            marker=dict(colors=["#ef4444", "#10b981"]),
            textinfo="percent+label",
            textfont=dict(size=13, color="#1e293b"),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(color="#1e293b"), showlegend=False,
            margin=dict(t=30, b=10, l=10, r=10), height=300,
            title=dict(text="Sentiment Split", font=dict(size=15, color="#1e293b")),
            annotations=[dict(
                text=f"<b>{pct_pos:.0f}%</b><br>Positive",
                x=0.5, y=0.5, font_size=18, font_color="#1e293b", showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    with c2:
        fig_hist = go.Figure()
        if not pos_df.empty:
            fig_hist.add_trace(go.Histogram(
                x=pos_df["confidence"], name="Positive",
                marker_color="#10b981", opacity=0.75, nbinsx=15,
            ))
        if not neg_df.empty:
            fig_hist.add_trace(go.Histogram(
                x=neg_df["confidence"], name="Negative",
                marker_color="#ef4444", opacity=0.75, nbinsx=15,
            ))
        fig_hist.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(color="#000000"), barmode="overlay",
            title=dict(text="Confidence Distribution", font=dict(size=15, color="#000000", family="Inter, sans-serif")),
            xaxis=dict(
                title=dict(text="Confidence Score", font=dict(size=13, color="#000000", family="Inter, sans-serif")),
                tickfont=dict(size=12, color="#000000", family="Inter, sans-serif"),
                gridcolor="#e2e8f0", zerolinecolor="#e2e8f0",
                linecolor="#000000", linewidth=1.5,
            ),
            yaxis=dict(
                title=dict(text="Count", font=dict(size=13, color="#000000", family="Inter, sans-serif")),
                tickfont=dict(size=12, color="#000000", family="Inter, sans-serif"),
                gridcolor="#e2e8f0", zerolinecolor="#e2e8f0",
                linecolor="#000000", linewidth=1.5,
            ),
            legend=dict(
                font=dict(size=12, color="#000000", family="Inter, sans-serif"),
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            ),
            margin=dict(t=50, b=10, l=10, r=10), height=300,
        )
        st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Side-by-side scrollable tweet panels ─────────────────────────────
    tc1, tc2 = st.columns(2)

    with tc1:
        # Green-tinted panel header
        st.markdown(
            f'<div class="panel-header-pos">🟢 Top Positive Tweets ({n_pos})</div>',
            unsafe_allow_html=True
        )
        # Scrollable container with green background applied via CSS
        with st.container(height=480, border=True):
            if pos_df.empty:
                st.markdown(
                    "<div style='color:#64748b;padding:1rem;text-align:center;"
                    "font-size:0.9rem;'>No positive tweets found.</div>",
                    unsafe_allow_html=True
                )
            else:
                # Inject tint background into this container
                st.markdown(
                    "<style>[data-testid='stVerticalBlockBorderWrapper']:first-of-type"
                    "{background:#f0fdf4 !important;border-color:#bbf7d0 !important;}</style>",
                    unsafe_allow_html=True
                )
                for _, row in pos_df.iterrows():
                    st.markdown(tweet_card_html(row.to_dict(), sentiment="pos"),
                                unsafe_allow_html=True)

    with tc2:
        # Red-tinted panel header
        st.markdown(
            f'<div class="panel-header-neg">🔴 Top Negative Tweets ({n_neg})</div>',
            unsafe_allow_html=True
        )
        with st.container(height=480, border=True):
            if neg_df.empty:
                st.markdown(
                    "<div style='color:#64748b;padding:1rem;text-align:center;"
                    "font-size:0.9rem;'>No negative tweets found.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<style>[data-testid='stVerticalBlockBorderWrapper']:last-of-type"
                    "{background:#fff5f5 !important;border-color:#fecaca !important;}</style>",
                    unsafe_allow_html=True
                )
                for _, row in neg_df.iterrows():
                    st.markdown(tweet_card_html(row.to_dict(), sentiment="neg"),
                                unsafe_allow_html=True)

    # Inject panel background colors more robustly via JS-free CSS targeting
    st.markdown("""
    <style>
    /* Target the two scrollable containers by order inside the two-column layout */
    .block-container section[data-testid="stVerticalBlock"] > div:nth-child(1)
        [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #f0fdf4 !important;
        border-color: #bbf7d0 !important;
    }
    .block-container section[data-testid="stVerticalBlock"] > div:nth-child(2)
        [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #fff5f5 !important;
        border-color: #fecaca !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Word clouds ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">☁️ Word Clouds</div>', unsafe_allow_html=True)
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown('<div class="wc-panel-title">🌐 All Tweets</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="wc-panel">', unsafe_allow_html=True)
            make_wordcloud_html(df["text"].tolist(),      "Blues",  key="wc_all")
            st.markdown('</div>', unsafe_allow_html=True)
    with wc2:
        st.markdown('<div class="wc-panel-title">🟢 Positive Tweets</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="wc-panel">', unsafe_allow_html=True)
            if n_pos > 0:
                make_wordcloud_html(pos_df["text"].tolist(),  "Greens", key="wc_pos")
            else:
                st.info("No positive tweets for word cloud")
            st.markdown('</div>', unsafe_allow_html=True)
    with wc3:
        st.markdown('<div class="wc-panel-title">🔴 Negative Tweets</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="wc-panel">', unsafe_allow_html=True)
            if n_neg > 0:
                make_wordcloud_html(neg_df["text"].tolist(),  "Reds",   key="wc_neg")
            else:
                st.info("No negative tweets for word cloud")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Full table + CSV ──────────────────────────────────────────────────
    with st.expander("📋 Full Results Table"):
        disp = df[["text", "label", "confidence"]].copy()
        disp["confidence"] = disp["confidence"].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(disp, width="stretch", height=300)
        csv = df[["text", "label", "confidence", "pos_score", "neg_score"]
                 ].to_csv(index=False).encode("utf-8")
        fname = f"{keyword.replace(' ', '_')}_sentiment.csv"
        st.download_button("⬇️ Download CSV", csv, fname, "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    MODEL_PATH = st.text_input(
        "Model folder path",
        value=r".\best_ml_model",
        help="Folder containing classifier.pkl, count_vectorizer.pkl, tfidf_vectorizer.pkl, label_map.json"
    )
    MAX_TWEETS = st.slider("Max tweets to fetch", 10, 150, 50, 10)
    st.markdown("---")
    st.markdown("""
**One-time install:**
```bash
pip install streamlit selenium \\
  webdriver-manager scikit-learn \\
  scipy wordcloud matplotlib \\
  pandas numpy plotly pillow
```
**Run:**
```bash
streamlit run app.py
```
No API token needed.
Tweets fetched free via Nitter.
Only **Nepali-script** tweets are kept.
Model: Traditional ML (sklearn).
""")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HERO (below navbar)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="hero-title">Political Sentiment Analysis</div>

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
# Define 3 tabs instead of 2, putting Home first so it loads by default
tab_home, tab1, tab2 = st.tabs(["🏠 Home", "🔍 Search Twitter", "📝 Paste Text for Analysis"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — HOME / DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_home:
    home_html = """
<div class="home-hero-bg">

  <!-- ░░ Floating Nepali word cloud background ░░ -->
    <div class="word-cloud-bg">
  <span class="floating-word">भ्रष्टाचार</span>
  <span class="floating-word">राजनीति</span>
  <span class="floating-word">सुशासन</span>
  <span class="floating-word">सरकार</span>
  <span class="floating-word">दल</span>
  <span class="floating-word">चुनाव</span>
  <span class="floating-word">नेता</span>
  <span class="floating-word">संसद</span>
  <span class="floating-word">विपक्ष</span>
  <span class="floating-word">लोकतन्त्र</span>
  <span class="floating-word">जनता</span>
  <span class="floating-word">मतदान</span>
  <span class="floating-word">नीति</span>
  <span class="floating-word">अधिकार</span>
  <span class="floating-word">न्याय</span>
  <span class="floating-word">विकास</span>
  <span class="floating-word">राष्ट्र</span>
  <span class="floating-word">प्रजातन्त्र</span>
  <span class="floating-word">स्वतन्त्रता</span>
  <span class="floating-word">समाज</span>
  <span class="floating-word">शासन</span>
  <span class="floating-word">मन्त्री</span>
  <span class="floating-word">प्रधानमन्त्री</span>
  <span class="floating-word">संविधान</span>
  <span class="floating-word">आन्दोलन</span>
  <span class="floating-word">कानुन</span>
  <span class="floating-word">अदालत</span>
  <span class="floating-word">पार्टी</span>
  <span class="floating-word">गठबन्धन</span>
  <span class="floating-word">भोट</span>
</div>

  <!-- ░░ Hero content overlay ░░ -->
  <div class="home-hero-content">
    <div class="home-hero-title">Nepali Political<br><span>Sentiment Analysis</span></div>
    <div class="home-hero-subtitle">
      Automatically gauge public opinion on Nepali political keywords — live from Twitter via Nitter, or from text you paste directly.
    </div>


    <!-- feature mini-cards -->
    <div class="feature-row">
      <div class="feature-card">
        <span class="feature-icon">🐦</span>
        <div class="feature-label">Live Twitter Scraping<br><span style="opacity:.65;font-weight:400;">via Nitter · No API key</span></div>
      </div>
      <div class="feature-card">
        <span class="feature-icon">🧠</span>
        <div class="feature-label">Traditional ML Model<br><span style="opacity:.65;font-weight:400;">sklearn · fast &amp; lightweight</span></div>
      </div>
      <div class="feature-card">
        <span class="feature-icon">☁️</span>
        <div class="feature-label">Devanagari Word Clouds<br><span style="opacity:.65;font-weight:400;">Positive · Negative · All</span></div>
      </div>
      <div class="feature-card">
        <span class="feature-icon">📋</span>
        <div class="feature-label">Paste &amp; Analyze<br><span style="opacity:.65;font-weight:400;">Instant per-line results</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ░░ About / Project Details card ░░ -->
<div class="home-about-card">
  <h3 class="project-details-title">🛠️ Project Details</h3>

  <ul class="project-details-list">
    <li><strong>Model:</strong> Traditional Machine Learning pipeline using <code>scikit-learn</code> — CountVectorizer + TF-IDF features with a trained classifier (e.g. Logistic Regression / SVM).</li>
    <li><strong>Dataset:</strong> 2022 election based political tweets</li>
    <li><strong>Labels:</strong> Two classes — <strong>Positive</strong> and <strong>Negative</strong>.</li>
    <li><strong>Confidence:</strong> High ≥ 80%, Medium 60–80%, Low &lt; 60% with review guidance.</li>
    <li><strong>Language:</strong> Nepali only.</li>
  </ul>

  <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:0.9rem 1.1rem;font-size:0.88rem;color:#92400e;">
    <strong>⚠️ Language constraint:</strong> This system strictly processes <strong>Nepali (Devanagari)</strong> text only. English and non-Devanagari tweets are automatically filtered out.
  </div>

  <div style="margin-top:1.5rem;text-align:center;padding:1rem;background:linear-gradient(135deg,#eff6ff,#f0fdf4);border-radius:12px;">
    <div style="font-size:0.9rem;color:#475569;font-weight:600;">
      <span class="live-dot"></span>Ready · Select a tab above to start analyzing
    </div>
  </div>
</div>
"""
    st.markdown(home_html.replace('\n', ' '), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TWITTER SEARCH
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # Title and description pulled upwards
    st.markdown("""
    <div style="margin-top: -0.5rem; margin-bottom: 1.5rem;">
        <h2 style="color:#0f172a; font-weight: 800; font-size: 1.8rem; margin-top: 0; margin-bottom: 0.4rem;">
            Twitter Sentiment Analysis
        </h2>
        <p style="color:#475569; font-size:1.05rem; margin:0; line-height: 1.5;">
            Search for political keywords and analyze the sentiment in Nepali tweets. Only Nepali-language tweets are fetched and analyzed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        keyword = st.text_input(
            "Search keyword", key="twitter_search", label_visibility="collapsed",
            placeholder="e.g. KP Oli · Balen · प्रधानमन्त्री · Congress · एमाले"
        )
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Tweets", key="analyze_twitter",
                                    width="stretch")

    if analyze_clicked and keyword.strip():
        with st.spinner("Loading model…"):
            try:
                clf, count_vec, tfidf_vec, label_map = load_model(MODEL_PATH)
            except Exception as e:
                st.error(f"❌ Could not load model from `{MODEL_PATH}`\n\n{e}")
                st.stop()

        status_box = st.empty()
        status_box.info(f"🌐 Scraping Nepali tweets for **{keyword}**…")
        raw_tweets = scrape_nitter(keyword=keyword, target=MAX_TWEETS, status_box=status_box)

        if not raw_tweets:
            status_box.error(
                "❌ No Nepali tweets fetched.\n"
                "- All Nitter instances may be down\n"
                "- No recent Nepali tweets for this keyword"
            )
            st.stop()

        status_box.success(f"✅ Fetched **{len(raw_tweets)} Nepali tweets** for **{keyword}**")

        with st.spinner(f"Analyzing {len(raw_tweets)} tweets…"):
            texts      = [t["text"] for t in raw_tweets]
            urls       = [t["url"]  for t in raw_tweets]
            sentiments = predict_sentiment_batch(texts, clf, count_vec, tfidf_vec, label_map)

        df = pd.DataFrame({
            "text"      : texts,
            "url"       : urls,
            "label"     : [s["label"]      for s in sentiments],
            "confidence": [s["confidence"] for s in sentiments],
            "pos_score" : [s["pos_score"]  for s in sentiments],
            "neg_score" : [s["neg_score"]  for s in sentiments],
        })

        display_results(df, keyword=keyword)

    elif analyze_clicked:
        st.warning("Please enter a keyword to search.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PASTE TEXT
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-bottom:0.75rem;'>"
        "Paste Nepali text below. Each line is analyzed separately.</p>",
        unsafe_allow_html=True
    )

    pasted_text = st.text_area(
        "Paste Nepali Text", height=150, label_visibility="collapsed",
        placeholder="Example:\nयो सरकारले राम्रो काम गरेको छ\nभ्रष्टाचारले देश सकियो",
    )

    col_pbtn, _ = st.columns([2, 5])
    with col_pbtn:
        analyze_pasted = st.button("🔍 Read Full Analysis", key="analyze_pasted",
                                   width="stretch")

    if analyze_pasted and pasted_text.strip():
        lines = [line.strip() for line in pasted_text.split('\n') if line.strip()]
        if not lines:
            st.warning("Please paste at least one line to analyze.")
        else:
            with st.spinner("Loading model…"):
                try:
                    clf, count_vec, tfidf_vec, label_map = load_model(MODEL_PATH)
                except Exception as e:
                    st.error(f"❌ Could not load model from `{MODEL_PATH}`\n\n{e}")
                    st.stop()
            with st.spinner(f"Analyzing {len(lines)} line(s)…"):
                df_pasted = analyze_pasted_texts(lines, clf, count_vec, tfidf_vec, label_map)
            # Use the dedicated pasted-text view: verdict banner + word clouds only
            display_pasted_results(df_pasted)

    elif analyze_pasted:
        st.warning("Please paste some text to analyze.")