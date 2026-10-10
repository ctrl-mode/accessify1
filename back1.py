import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import csv

# ==========================================
# PAGE CONFIGURATION
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

    # --- Tabular Model (Risk Predictor using Real CSV Data) ---
    rf_features = []
    rf_labels = []

    try:
        # Pure Python CSV parsing (No Pandas)
        url_stats = {}
        with open("Original_full_data (1)_2.csv", mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('webURL', '')
                desc = str(row.get('Description', '')).lower()
                try:
                    impact = int(row.get('initialImpactScore', 0))
                except ValueError:
                    impact = 0
                
                if url not in url_stats:
                    url_stats[url] = {'missing_alts': 0, 'empty_links': 0, 'unlabeled_buttons': 0, 'max_impact': 0}
                
                # Track the highest severity score on this webpage
                url_stats[url]['max_impact'] = max(url_stats[url]['max_impact'], impact)
                
                # Map dataset text to our features
                if "alternate text" in desc:
                    url_stats[url]['missing_alts'] += 1
                if "links have discernible text" in desc:
                    url_stats[url]['empty_links'] += 1
                if "buttons have discernible text" in desc:
                    url_stats[url]['unlabeled_buttons'] += 1
                    
        # Build the final training arrays
        for url, stats in url_stats.items():
            rf_features.append([stats['missing_alts'], stats['empty_links'], stats['unlabeled_buttons']])
            # If the max impact score is 5, classify the page as High Risk (Fail = 1)
            rf_labels.append(1 if stats['max_impact'] >= 5 else 0)
            
    except FileNotFoundError:
        # Fail-safe: Uses synthetic data if the CSV is missing from GitHub
        rf_features = [[0, 0, 0], [1, 2, 0], [5, 10, 2], [12, 24, 5], [20, 30, 10]]
        rf_labels = [0, 0, 1, 1, 1] 

    # Train the Random Forest on the real dataset
    rf_model = RandomForestClassifier(random_state=42).fit(rf_features, rf_labels)
    
    return vectorizer, nlp_model, rf_model

vectorizer, nlp_model, rf_model = train_models()

# ==========================================
# 2. SIDEBAR: LIVE AI SANDBOX
# ==========================================
with st.sidebar:
    st.header("🧪 Live AI Sandbox")
    st.markdown("Test our NLP engine in real-time. Type an alt-text below to see if the AI flags it as **Junk**.")
    
    test_alt = st.text_input("Enter custom alt-text:", placeholder="e.g., 'image.png' or 'dog'")
    if test_alt:
        pred = nlp_model.predict(vectorizer.transform([test_alt]))[0]
        if test_alt.strip() == "" or pred == 0:
            st.error("🚫 AI Verdict: JUNK (Fails WCAG)")
        else:
            st.success("✅ AI Verdict: DESCRIPTIVE (Passes WCAG)")
            
    st.markdown("---")
    st.caption("Accessify Mk1 • Built for Hackathon")
    st.caption("Risk Engine Trained on Dataset: Original_full_data (1)_2.csv")

# ==========================================
# 3. THE MAIN DASHBOARD UI
# ==========================================
st.title("♿ Accessify: AI Accessibility Auditor")
st.markdown("Move beyond simple rule-based scanning. Use machine learning to detect **junk alt-text** and predict overall **compliance risk**.")

url_input = st.text_input("Enter Target URL", value="https://en.wikipedia.org/wiki/Main_Page")

if st.button("Run Full AI Audit 🚀", type="primary"):
    
    # --- ANIMATED PROGRESS BAR ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.markdown("🔍 **Step 1/3:** Scraping DOM elements...")
        progress_bar.progress(30)
        time.sleep(0.5)

        # --- THE SCRAPER ---
        response = requests.get(url_input, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        all_alt_texts = [img.get("alt", "") for img in soup.find_all("img") if img.get("alt") is not None]
        missing_alts = len([img for img in soup.find_all("img") if not img.get("alt")])
        empty_links = len([a for a in soup.find_all("a") if not a.text.strip()])
        unlabeled_buttons = len([b for b in soup.find_all("button") if not b.text.strip() and not b.get("aria-label")])

        status_text.markdown("🧠 **Step 2/3:** Running NLP Engine on Alt-Texts...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        status_text.markdown("⚙️ **Step 3/3:** Calculating Overall Compliance Risk...")
        progress_bar.progress(90)
        time.sleep(0.5)

        # --- PREDICT OVERALL RISK ---
        live_data = [[missing_alts, empty_links, unlabeled_buttons]]
        risk_prediction = rf_model.predict(live_data)[0]
        risk_probability = rf_model.predict_proba(live_data)[0][1] * 100
        
        progress_bar.progress(100)
        status_text.empty() # Clear the status text
        st.toast('Audit Completed Successfully!', icon='🎉')

        # ==========================================
        # 4. TABBED RESULTS DISPLAY
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🧠 Explainable AI", "📝 NLP Breakdown"])

        # --- TAB 1: DASHBOARD ---
        with tab1:
            st.markdown("### Structural DOM Diagnostics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Missing Alt Tags", missing_alts)
            col2.metric("Empty Links", empty_links)
            col3.metric("Unlabeled Buttons", unlabeled_buttons)
            
            if risk_prediction == 1:
                col4.error(f"FAIL RISK: {risk_probability:.1f}%")
            else:
                col4.success(f"PASS PROBABILITY: {100 - risk_probability:.1f}%")
                st.balloons() # Trigger celebration effect if the site passes

        # --- TAB 2: EXPLAINABLE AI ---
        with tab2:
            st.markdown("### Decision Weights")
            st.caption("Which structural errors are driving this website's risk score? (Higher bar = Greater risk impact)")
            
            importances = rf_model.feature_importances_
            
            # Pure Python dictionary replacing Pandas
            chart_data = {
                "Impact Weight": {
                    "Missing Alts": importances[0],
                    "Empty Links": importances[1],
                    "Unlabeled Buttons": importances[2]
                }
            }
            st.bar_chart(chart_data)

        # --- TAB 3: NLP BREAKDOWN ---
        with tab3:
            st.markdown("### 'Junk Alt-Text' Detection")
            
            if not all_alt_texts:
                st.info("No alt-text attributes found to analyse on this page.")
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
                
                with st.expander("View Passed Alt-Texts (Valid Descriptions)"):
                    for text in all_alt_texts:
                        if text.strip() != "" and vectorizer.transform([text]).sum() > 0:
                            st.write(f"- {text}")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Failed to scan website. The site might be blocking web scrapers. Error: {e}")
