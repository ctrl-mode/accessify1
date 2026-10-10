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
st.set_page_config(page_title="Accessify Mk1", page_icon="⚡", layout="wide")

# ==========================================
# INJECT PREMIUM DARK UI (NO GLOW)
# ==========================================
st.html(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --ax-bg: #080b12;
        --ax-text: #eef3ff;
        --ax-muted: #8f9bb4;
        --ax-line: rgba(150, 170, 205, .14);
      }
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .stApp { background: var(--ax-bg); color: var(--ax-text); }
      
      /* Clean, flat metric cards without the glowing box-shadow */
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20,28,44,.91), rgba(13,18,30,.88));
        padding: 17px 18px;
        border: 1px solid var(--ax-line);
        border-radius: 12px;
      }
      [data-testid="stMetricLabel"] { color: #9ca9c2; font-size: .78rem; }
      [data-testid="stMetricValue"] { color: #f3f6ff; font-weight: 800; }
      
      /* Hero banner without neon shadows */
      .ax-hero {
        padding:28px 30px 27px; 
        border-radius:16px;
        border:1px solid rgba(134,158,206,.2);
        background: linear-gradient(130deg, rgba(22,31,51,.96), rgba(13,18,31,.94));
        margin-bottom:20px;
      }
      .ax-pill { 
        display:inline-flex; align-items:center; gap:7px; padding:6px 10px; 
        border:1px solid rgba(105,240,228,.2); border-radius:999px; color:#9af4e5; 
        background:rgba(105,240,228,.06); font:10px 'DM Mono',monospace; letter-spacing:1px; 
      }
      .ax-dot { height:7px; width:7px; border-radius:50%; background:#75e8bd; display:inline-block; }
      .ax-title { font-size:clamp(27px, 3.2vw, 42px); line-height:1.1; font-weight:850; letter-spacing:-1.7px; margin:11px 0 10px; color:#f5f7ff; }
      .ax-subtitle { color:#a2aec6; font-size:14px; line-height:1.7; max-width:690px; margin:0; }
      
      /* Typography */
      .ax-brand { display:flex; align-items:center; gap:11px; margin: 3px 0 24px; }
      .ax-mark {
        display:grid; place-items:center; width:42px; height:42px; border-radius:10px;
        background: linear-gradient(135deg, #72f4e5, #7788ff);
        color:#07111a; font-size:22px; font-weight:900;
      }
      .ax-brandname { font-size:19px; font-weight:850; letter-spacing:-.8px; color:#f0f4ff; }
      .ax-brandline { font:10px 'DM Mono', monospace; letter-spacing:1.65px; color:#8997b4; margin-top:2px; }
      .ax-eyebrow { color:#75eadd; font: 11px 'DM Mono', monospace; letter-spacing:2px; text-transform:uppercase; margin-bottom: 10px;}
    </style>
    """
)

# ==========================================
# 1. TRAIN THE AI MODELS (Pure Python, No Pandas)
# ==========================================
@st.cache_resource
def train_models():
    # --- NLP Model (Junk Alt-Text) ---
    nlp_texts = ["...", " ", "image", "image.png", "img", "picture", "logo.jpg", 
                 "a golden retriever catching a red frisbee", 
                 "bar chart showing quarter three sales growth", 
                 "student walking across the university campus"]
    nlp_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] 

    vectorizer = TfidfVectorizer()
    X_nlp = vectorizer.fit_transform(nlp_texts)
    nlp_model = LogisticRegression().fit(X_nlp, nlp_labels)

    # --- Tabular Model (Trained on CSV without Pandas) ---
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
        # Failsafe if CSV isn't found
        rf_features = [[0, 0, 0], [1, 2, 0], [5, 10, 2], [12, 24, 5], [20, 30, 10]]
        rf_labels = [0, 0, 1, 1, 1] 

    rf_model = RandomForestClassifier(random_state=42).fit(rf_features, rf_labels)
    return vectorizer, nlp_model, rf_model

vectorizer, nlp_model, rf_model = train_models()

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(
        '<div class="ax-brand"><div class="ax-mark">A</div><div>'
        '<div class="ax-brandname">accessify<span style="color:#6be9dd">.</span></div>'
        '<div class="ax-brandline">AUTOMATION ENGINE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="ax-eyebrow">🧪 MANUAL NLP SANDBOX</div>', unsafe_allow_html=True)
    st.caption("Override the DOM scanner. Test single alt-text strings against the neural net.")
    
    test_alt = st.text_input("Input String:", placeholder="e.g., 'image.png' or 'dog'")
    if test_alt:
        pred = nlp_model.predict(vectorizer.transform([test_alt]))[0]
        if test_alt.strip() == "" or pred == 0:
            st.error("🚫 VERDICT: JUNK STRING")
        else:
            st.success("✅ VERDICT: VALIDATED")
            
    st.markdown("---")
    st.caption("DATABANKS: Original_full_data (1)_2.csv")

# ==========================================
# 3. MAIN DASHBOARD UI
# ==========================================
st.markdown(
    '<div class="ax-hero"><span class="ax-pill"><span class="ax-dot"></span>LIVE AUDITOR</span>'
    '<div class="ax-title">Accessify Protocol</div>'
    '<p class="ax-subtitle">Automated Neural Accessibility Engine. Input target mainframe URL below to commence scanning.</p></div>',
    unsafe_allow_html=True,
)

url_input = st.text_input("Target URL", value="https://en.wikipedia.org/wiki/Main_Page")

if st.button("Initialize Engine 🚀", type="primary"):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.markdown("🔍 **Step 1:** Extracting DOM elements and line attributes...")
        progress_bar.progress(30)
        time.sleep(0.5)

        # --- THE SCRAPER ---
        response = requests.get(url_input, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        # Using html.parser allows us to extract line numbers via .sourceline
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract images and their HTML line numbers
        image_nodes = soup.find_all("img")
        missing_alts = 0
        alt_data = [] # List of dicts containing the text and the line number
        
        for img in image_nodes:
            alt = img.get("alt")
            # Grab the exact line number from the parsed HTML document
            line = getattr(img, 'sourceline', 'Unknown')
            
            if alt is None:
                missing_alts += 1
            else:
                alt_data.append({"text": alt, "line": line})

        empty_links = len([a for a in soup.find_all("a") if not a.text.strip()])
        unlabeled_buttons = len([b for b in soup.find_all("button") if not b.text.strip() and not b.get("aria-label")])

        status_text.markdown("🧠 **Step 2:** Running NLP Engine on extracted attributes...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        status_text.markdown("⚙️ **Step 3:** Calculating Risk Compliance...")
        progress_bar.progress(90)
        time.sleep(0.5)

        # --- PREDICT OVERALL RISK ---
        live_data = [[missing_alts, empty_links, unlabeled_buttons]]
        risk_prediction = rf_model.predict(live_data)[0]
        risk_probability = rf_model.predict_proba(live_data)[0][1] * 100
        
        progress_bar.progress(100)
        status_text.empty() 
        st.toast('Engine Audit Complete', icon='✅')

        # ==========================================
        # 4. TABBED RESULTS DISPLAY
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🧠 Explainable AI", "📝 Semantic Logs"])

        with tab1:
            st.markdown("### Structural Diagnostics")
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
            st.markdown("### Decision Weights")
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
            st.markdown("### NLP Threat Detection")
            if not alt_data:
                st.info("No alt-text nodes found in target DOM.")
            else:
                # Prepare data for prediction
                texts_to_predict = [d["text"] for d in alt_data]
                X_live = vectorizer.transform(texts_to_predict)
                predictions = nlp_model.predict(X_live)

                junk_found = False
                # Pair the text, the specific line number, and the AI prediction
                for data, prediction in zip(alt_data, predictions):
                    text = data["text"]
                    line = data["line"]
                    
                    if text.strip() == "":
                        # Explicitly format the error just as requested
                        st.error(f'🚫 **Lazy code in {line} line:** " " (Empty String Detected)')
                        junk_found = True
                    elif prediction == 0:
                        st.error(f'🚫 **AI Flagged Junk in {line} line:** "{text}"')
                        junk_found = True
                
                if not junk_found:
                    st.success("✅ NLP Engine Clear: All alt-texts are semantically valid.")
                
                with st.expander("View Passed Alt-Texts"):
                    for data in alt_data:
                        text = data["text"]
                        line = data["line"]
                        if text.strip() != "" and vectorizer.transform([text]).sum() > 0:
                            st.write(f"- Line {line}: `{text}`")

    except Exception as e:
        st.error(f"Engine connection failed. The site might be blocking web scrapers. Error: {e}")
