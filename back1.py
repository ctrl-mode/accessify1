import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import csv

# ==========================================
# PAGE CONFIGURATION (FUTURISTIC WIDE)
# ==========================================
st.set_page_config(page_title="Accessify Mk1", page_icon="⚡", layout="wide")

# ==========================================
# INJECT CYBERPUNK CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace !important;
    }
    
    .glow-title {
        color: #00ffcc;
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #0099ff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0px;
    }
    
    .cyber-sub {
        color: #a8b2c1;
        font-size: 1.2rem;
        margin-bottom: 20px;
        border-left: 3px solid #00ffcc;
        padding-left: 10px;
    }
    
    .metric-card {
        border: 1px solid #00ffcc;
        background: rgba(0, 255, 204, 0.05);
        border-radius: 4px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. TRAIN THE AI MODELS
# ==========================================
@st.cache_resource
def train_models():
    nlp_texts = ["...", " ", "image", "image.png", "img", "picture", "logo.jpg", 
                 "a golden retriever catching a red frisbee", 
                 "bar chart showing quarter three sales growth", 
                 "student walking across the university campus"]
    nlp_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    vectorizer = TfidfVectorizer()
    X_nlp = vectorizer.fit_transform(nlp_texts)
    nlp_model = LogisticRegression().fit(X_nlp, nlp_labels)

    rf_features = []
    rf_labels = []

    try:
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
                url_stats[url]['max_impact'] = max(url_stats[url]['max_impact'], impact)
                
                if "alternate text" in desc: url_stats[url]['missing_alts'] += 1
                if "links have discernible text" in desc: url_stats[url]['empty_links'] += 1
                if "buttons have discernible text" in desc: url_stats[url]['unlabeled_buttons'] += 1
                    
        for url, stats in url_stats.items():
            rf_features.append([stats['missing_alts'], stats['empty_links'], stats['unlabeled_buttons']])
            rf_labels.append(1 if stats['max_impact'] >= 5 else 0)
            
    except FileNotFoundError:
        rf_features = [[0, 0, 0], [1, 2, 0], [5, 10, 2], [12, 24, 5], [20, 30, 10]]
        rf_labels = [0, 0, 1, 1, 1] 

    rf_model = RandomForestClassifier(random_state=42).fit(rf_features, rf_labels)
    return vectorizer, nlp_model, rf_model

vectorizer, nlp_model, rf_model = train_models()

# ==========================================
# 2. SIDEBAR: COMMAND CENTER
# ==========================================
with st.sidebar:
    st.markdown("<h2 class='glow-title' style='font-size: 1.8rem;'>⚡ COMMAND_CENTER</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧪 MANUAL NLP INJECTION")
    st.caption("Override scanner. Test single alt-text strings against the neural net.")
    
    test_alt = st.text_input(">>> INPUT_STRING:", placeholder="e.g., 'image.png' or 'dog'")
    if test_alt:
        pred = nlp_model.predict(vectorizer.transform([test_alt]))[0]
        if test_alt.strip() == "" or pred == 0:
            st.error("⚠️ FATAL: JUNK STRING DETECTED")
        else:
            st.success("✅ VERIFIED: SEMANTICALLY SOUND")
            
    st.markdown("---")
    st.caption("SYSTEM STATUS: ONLINE")
    st.caption("DATABANKS: Original_full_data (1)_2.csv")

# ==========================================
# 3. MAIN UI & HACKER TERMINAL ANIMATION
# ==========================================
st.markdown("<h1 class='glow-title'>ACCESSIFY // PROTOCOL v1.0</h1>", unsafe_allow_html=True)
st.markdown("<div class='cyber-sub'>AUTOMATED NEURAL ACCESSIBILITY AUDITOR</div>", unsafe_allow_html=True)

url_input = st.text_input(">>> TARGET_MAINFRAME_URL", value="https://en.wikipedia.org/wiki/Main_Page")

if st.button("EXECUTE // AUDIT_SEQUENCE 🚀", type="primary"):
    
    # --- TERMINAL BOOTUP ANIMATION ---
    terminal_display = st.empty()
    logs = "> INITIALIZING ACCESSIFY PROTOCOL...\n"
    terminal_display.code(logs, language="bash")
    time.sleep(0.4)
    
    try:
        logs += f"> PINGING {url_input}...\n"
        terminal_display.code(logs, language="bash")
        time.sleep(0.4)

        # --- THE SCRAPER ---
        response = requests.get(url_input, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        all_alt_texts = [img.get("alt", "") for img in soup.find_all("img") if img.get("alt") is not None]
        missing_alts = len([img for img in soup.find_all("img") if not img.get("alt")])
        empty_links = len([a for a in soup.find_all("a") if not a.text.strip()])
        unlabeled_buttons = len([b for b in soup.find_all("button") if not b.text.strip() and not b.get("aria-label")])

        logs += f"> DOM EXTRACTED: {len(all_alt_texts)} IMAGE NODES LOCATED.\n"
        terminal_display.code(logs, language="bash")
        time.sleep(0.5)

        logs += "> RUNNING NATURAL LANGUAGE PROCESSING VECTORS...\n"
        terminal_display.code(logs, language="bash")
        time.sleep(0.5)
        
        logs += "> CALCULATING RANDOM FOREST COMPLIANCE MATRICES...\n"
        terminal_display.code(logs, language="bash")
        time.sleep(0.6)

        # --- PREDICT OVERALL RISK ---
        live_data = [[missing_alts, empty_links, unlabeled_buttons]]
        risk_prediction = rf_model.predict(live_data)[0]
        risk_probability = rf_model.predict_proba(live_data)[0][1] * 100
        
        logs += "> AUDIT COMPLETE. RENDERING DASHBOARD...\n"
        terminal_display.code(logs, language="bash")
        time.sleep(0.5)
        
        terminal_display.empty() # Clear terminal to reveal the clean dashboard
        st.toast('SYSTEM COMPROMISE ASSESSMENT COMPLETE', icon='⚡')

        # ==========================================
        # 4. TABBED CYBER RESULTS
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📊 SYSTEM_OVERVIEW", "🧠 NEURAL_PATHWAYS", "📝 SEMANTIC_LOGS"])

        with tab1:
            st.markdown("### 📡 STRUCTURAL DIAGNOSTICS")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Missing Alt Tags", missing_alts, "- Critical", delta_color="inverse")
            col2.metric("Empty Links", empty_links, "- Warning", delta_color="inverse")
            col3.metric("Unlabeled Buttons", unlabeled_buttons, "- Warning", delta_color="inverse")
            
            if risk_prediction == 1:
                col4.error(f"FAIL RISK: {risk_probability:.1f}%")
            else:
                col4.success(f"PASS PROBABILITY: {100 - risk_probability:.1f}%")
                st.balloons()

        with tab2:
            st.markdown("### 🧠 ALGORITHM DECISION WEIGHTS")
            st.caption("Visualizing the Random Forest threat-assessment matrix.")
            importances = rf_model.feature_importances_
            chart_data = {
                "Threat Impact": {
                    "Missing Alts": importances[0],
                    "Empty Links": importances[1],
                    "Unlabeled Buttons": importances[2]
                }
            }
            st.bar_chart(chart_data)

        with tab3:
            st.markdown("### 📝 NLP THREAT DETECTION")
            if not all_alt_texts:
                st.info("No alt-text nodes found in target DOM.")
            else:
                X_live = vectorizer.transform(all_alt_texts)
                predictions = nlp_model.predict(X_live)

                junk_found = False
                for text, prediction in zip(all_alt_texts, predictions):
                    if text.strip() == "":
                        st.error('🚫 **LAZY CODE:** " " (Empty String Detected)')
                        junk_found = True
                    elif prediction == 0:
                        st.error(f'🚫 **AI FLAGGED JUNK:** "{text}"')
                        junk_found = True
                
                if not junk_found:
                    st.success("✅ NLP ENGINE CLEAR: All alt-texts are semantically valid.")
                
                with st.expander("VIEW SECURED LOGS (Passed Alt-Texts)"):
                    for text in all_alt_texts:
                        if text.strip() != "" and vectorizer.transform([text]).sum() > 0:
                            st.write(f"> {text}")

    except Exception as e:
        terminal_display.empty()
        st.error(f"SYSTEM OVERLOAD. Target mainframe rejected connection. Error: {e}")
