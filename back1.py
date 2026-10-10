import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time

# ==========================================
# PAGE CONFIGURATION (Makes it look professional)
# ==========================================
st.set_page_config(page_title="Accessify Mk1", page_icon="♿", layout="wide")

# ==========================================
# 1. TRAIN THE AI MODELS (Cached for speed)
# ==========================================
@st.cache_resource
def train_models():
    # --- NLP Model (Junk Alt-Text) ---
    nlp_texts = ["...", " ", "image", "image.png", "img", "picture", "logo.jpg", 
                "a golden retriever catching a red frisbee", 
                "bar chart showing quarter three sales growth", 
                "student walking across the university campus"]
    nlp_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] # 0 = Junk, 1 = Good

    vectorizer = TfidfVectorizer()
    X_nlp = vectorizer.fit_transform(nlp_texts)
    nlp_model = LogisticRegression().fit(X_nlp, nlp_labels)

    # --- Tabular Model (Risk Predictor) ---
    # Synthetic data: [Missing Alts, Empty Links, Unlabeled Buttons]
    rf_features = [[0, 0, 0], [1, 2, 0], [5, 10, 2], [12, 24, 5], [20, 30, 10]]
    rf_labels = [0, 0, 1, 1, 1] # 0 = Pass, 1 = Fail

    rf_model = RandomForestClassifier(random_state=42).fit(rf_features, rf_labels)
    
    return vectorizer, nlp_model, rf_model

vectorizer, nlp_model, rf_model = train_models()

# ==========================================
# 2. THE DASHBOARD UI
# ==========================================
st.title("♿ Accessify: AI Accessibility Auditor")
st.markdown("Move beyond simple rule-based scanning. Use machine learning to detect **junk alt-text** and predict overall **compliance risk**.")

url_input = st.text_input("Enter Target URL", value="https://en.wikipedia.org/wiki/Main_Page")

if st.button("Run Full AI Audit", type="primary"):
    with st.spinner(f"Scraping and analyzing {url_input} ..."):
        try:
            # Add a slight delay for dramatic effect during the demo
            time.sleep(1)

            # --- THE SCRAPER ---
            response = requests.get(url_input, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            all_alt_texts = [img.get("alt", "") for img in soup.find_all("img") if img.get("alt") is not None]
            missing_alts = len([img for img in soup.find_all("img") if not img.get("alt")])
            empty_links = len([a for a in soup.find_all("a") if not a.text.strip()])
            unlabeled_buttons = len([b for b in soup.find_all("button") if not b.text.strip() and not b.get("aria-label")])

            st.success("DOM Extraction and AI Analysis Complete!")

            # --- PREDICT OVERALL RISK ---
            live_data = [[missing_alts, empty_links, unlabeled_buttons]]
            risk_prediction = rf_model.predict(live_data)[0]
            risk_probability = rf_model.predict_proba(live_data)[0][1] * 100

            # --- DISPLAY STRUCTURAL METRICS ---
            st.markdown("### 📊 Structural DOM Diagnostics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Missing Alt Tags", missing_alts)
            col2.metric("Empty Links", empty_links)
            col3.metric("Unlabeled Buttons", unlabeled_buttons)
            
            if risk_prediction == 1:
                col4.error(f"FAIL RISK: {risk_probability:.1f}%")
            else:
                col4.success(f"PASS PROBABILITY: {100 - risk_probability:.1f}%")

            # --- DISPLAY EXPLAINABLE AI (XAI) ---
            st.markdown("### 🧠 Explainable AI (XAI) Weights")
            st.caption("Which structural errors are driving the risk score?")
            importances = rf_model.feature_importances_
            feature_names = ["Missing Alts", "Empty Links", "Unlabeled Buttons"]
            chart_data = pd.DataFrame(importances, index=feature_names, columns=["Impact Weight"])
            st.bar_chart(chart_data)

            # --- DISPLAY NLP TEXT ANALYSIS ---
            st.markdown("### 📝 NLP 'Junk Alt-Text' Detection")
            
            if not all_alt_texts:
                st.info("No alt-text attributes found to analyze.")
            else:
                X_live = vectorizer.transform(all_alt_texts)
                predictions = nlp_model.predict(X_live)

                junk_found = False
                for text, prediction in zip(all_alt_texts, predictions):
                    if text.strip() == "":
                        st.error('🚫 **Detected Empty String:** " " (Junk)')
                        junk_found = True
                    elif prediction == 0:
                        st.error(f'🚫 **Flagged by AI:** "{text}" (Junk)')
                        junk_found = True
                
                if not junk_found:
                    st.success("✅ All provided alt-texts passed the NLP quality check.")
                    with st.expander("View Passed Alt-Texts"):
                        for text in all_alt_texts:
                            if text.strip() != "":
                                st.write(f"- {text}")

        except Exception as e:
            st.error(f"Failed to scan website. Please check the URL or try another site. Error: {e}")