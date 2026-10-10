from __future__ import annotations

import ipaddress
import socket
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =============================================================================
# ACCESSIFY — AUTOMATED WEBSITE ACCESSIBILITY AUDITOR
# =============================================================================
st.set_page_config(
    page_title="Accessify | Accessibility Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ax-bg:#080b12; --ax-line:rgba(150,170,205,.14); --ax-text:#eef3ff; --ax-muted:#8f9bb4; --ax-cyan:#69f0e4; --ax-blue:#7c8cff; }
      html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
      .stApp { background:radial-gradient(ellipse at 10% 0%,rgba(54,78,164,.22),transparent 36%),radial-gradient(ellipse at 92% 14%,rgba(25,174,164,.11),transparent 28%),#080b12; color:var(--ax-text); }
      [data-testid="stHeader"] { background:rgba(8,11,18,.72); }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(15,20,33,.98),rgba(9,12,21,.98)); border-right:1px solid var(--ax-line); }
      [data-testid="stSidebar"] > div { padding-top:1.1rem; }
      .block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1550px; }
      h1,h2,h3,h4 { letter-spacing:-.035em; } h1 { font-weight:850 !important; } h2,h3 { font-weight:780 !important; }
      p,label,.stMarkdown { color:var(--ax-text); }
      [data-testid="stCaptionContainer"] { color:var(--ax-muted); }
      [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(20,28,44,.91),rgba(13,18,30,.88)); padding:17px 18px; border:1px solid var(--ax-line); border-radius:17px; box-shadow:0 12px 35px rgba(0,0,0,.12); }
      [data-testid="stMetricLabel"] { color:#9ca9c2; font-size:.78rem; } [data-testid="stMetricValue"] { color:#f3f6ff; font-weight:800; }
      [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(18,25,40,.82),rgba(12,16,27,.86)); border-color:var(--ax-line)!important; border-radius:18px!important; }
      div[data-testid="stForm"] { border:1px solid var(--ax-line); border-radius:18px; padding:1.1rem; background:rgba(15,21,35,.54); }
      .stButton>button,.stDownloadButton>button { border-radius:11px; font-weight:750; border:1px solid rgba(126,148,191,.24); transition:all .18s ease; min-height:2.65rem; }
      .stButton>button:hover,.stDownloadButton>button:hover { border-color:rgba(105,240,228,.75); transform:translateY(-1px); box-shadow:0 5px 22px rgba(105,240,228,.08); }
      button[kind="primary"] { background:linear-gradient(110deg,#6be9dd,#91f3ce)!important; color:#07121a!important; border:0!important; }
      input,textarea,[data-baseweb="select"]>div { background-color:rgba(8,12,22,.86)!important; border-color:rgba(145,163,200,.2)!important; border-radius:10px!important; }
      [data-testid="stDataFrame"] { border:1px solid var(--ax-line); border-radius:12px; overflow:hidden; }
      [data-testid="stProgressBar"]>div>div { background:linear-gradient(90deg,#7587ff,#69f0e4); }
      hr { border-color:var(--ax-line)!important; }
      .ax-brand { display:flex; align-items:center; gap:11px; margin:3px 0 24px; }
      .ax-mark { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; background:linear-gradient(135deg,#72f4e5,#7788ff); color:#07111a; font-size:22px; font-weight:900; box-shadow:0 6px 24px rgba(105,240,228,.17); }
      .ax-brandname { font-size:19px; font-weight:850; letter-spacing:-.8px; color:#f0f4ff; }
      .ax-brandline { font:10px 'DM Mono',monospace; letter-spacing:1.65px; color:#8997b4; margin-top:2px; }
      .ax-eyebrow { color:#75eadd; font:11px 'DM Mono',monospace; letter-spacing:2px; text-transform:uppercase; }
      .ax-hero { position:relative; overflow:hidden; padding:29px 30px 28px; border-radius:23px; border:1px solid rgba(134,158,206,.2); background:radial-gradient(circle at 90% 10%,rgba(105,240,228,.13),transparent 24%),radial-gradient(circle at 72% 110%,rgba(124,140,255,.18),transparent 38%),linear-gradient(130deg,rgba(22,31,51,.96),rgba(13,18,31,.94)); box-shadow:0 24px 70px rgba(0,0,0,.18); margin-bottom:20px; }
      .ax-hero:after { content:''; position:absolute; right:-45px; top:-90px; width:260px; height:260px; border:1px solid rgba(105,240,228,.16); border-radius:50%; box-shadow:0 0 0 26px rgba(105,240,228,.025),0 0 0 54px rgba(124,140,255,.025); pointer-events:none; }
      .ax-title { font-size:clamp(27px,3.2vw,43px); line-height:1.1; font-weight:850; letter-spacing:-1.7px; margin:11px 0 10px; color:#f5f7ff; }
      .ax-gradient { background:linear-gradient(95deg,#72efe1,#a0b0ff); -webkit-background-clip:text; background-clip:text; color:transparent; }
      .ax-subtitle { color:#a2aec6; font-size:14px; line-height:1.7; max-width:760px; margin:0; }
      .ax-pill { display:inline-flex; align-items:center; gap:7px; padding:6px 10px; border:1px solid rgba(105,240,228,.2); border-radius:999px; color:#9af4e5; background:rgba(105,240,228,.06); font:10px 'DM Mono',monospace; letter-spacing:1px; }
      .ax-dot { height:7px; width:7px; border-radius:50%; background:#75e8bd; box-shadow:0 0 12px #75e8bd; display:inline-block; }
      .ax-section { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:17px 0 10px; }
      .ax-section-title { color:#f0f3ff; font-size:17px; font-weight:800; letter-spacing:-.5px; }
      .ax-section-note { color:#8290ab; font:10px 'DM Mono',monospace; letter-spacing:1px; }
      .ax-mini { font:10px 'DM Mono',monospace; color:#8290ab; letter-spacing:1.1px; }
      .ax-status { display:inline-flex; align-items:center; gap:6px; font:10px 'DM Mono',monospace; color:#81ecc9; background:rgba(129,236,201,.07); border:1px solid rgba(129,236,201,.17); border-radius:7px; padding:5px 8px; }
      .ax-callout { padding:13px 15px; border-radius:12px; background:rgba(105,240,228,.045); border:1px solid rgba(105,240,228,.16); color:#b8c9dd; font-size:12px; line-height:1.7; }
      .ax-list-row { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:11px 0; border-bottom:1px solid rgba(145,163,200,.10); }
      .ax-list-row:last-child { border-bottom:0; }
      .ax-list-title { font-weight:700; color:#e9eeff; font-size:12px; }
      .ax-list-sub { color:#8794ae; font-size:10px; margin-top:3px; }
      .ax-tag { border:1px solid rgba(135,153,190,.18); color:#a1aec8; border-radius:6px; padding:4px 7px; font:9px 'DM Mono',monospace; white-space:nowrap; }
      .ax-footer { margin-top:35px; padding-top:14px; border-top:1px solid rgba(145,163,200,.1); display:flex; justify-content:space-between; gap:10px; color:#65728c; font:9px 'DM Mono',monospace; letter-spacing:1px; }
      @keyframes axPulse { 0%,100%{opacity:.55} 50%{opacity:1} } .ax-live { animation:axPulse 2s ease-in-out infinite; }
      @media(max-width:700px) { .ax-hero{padding:22px 19px} .ax-title{letter-spacing:-1px} }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Session state -------------------------------------------------------------
if "ax_runs" not in st.session_state:
    st.session_state.ax_runs = []
if "ax_last_audit" not in st.session_state:
    st.session_state.ax_last_audit = None
if "ax_settings" not in st.session_state:
    st.session_state.ax_settings = {"timeout": 12, "show_terminal": True}


# --- Model training ------------------------------------------------------------
SEED_TEXTS = [
    "...",
    " ",
    "image",
    "image.png",
    "img",
    "picture",
    "logo.jpg",
    "photo",
    "IMG_1234.jpg",
    "a golden retriever catching a red frisbee",
    "bar chart showing quarter three sales growth",
    "student walking across the university campus",
    "a red car parked beside a tree",
    "a person using a wheelchair ramp outside a building",
    "close-up of hands typing on a laptop keyboard",
    "line graph comparing monthly revenue during 2024",
    "logo for Accessify website accessibility auditing software",
    "two children playing football in a grassy park",
    "a scientist adjusting a microscope in a laboratory",
    "screenshot of accessibility settings in a mobile app",
]
SEED_LABELS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
FALLBACK_RF_X = [[0, 0, 0], [1, 2, 0], [2, 4, 1], [5, 10, 2], [12, 24, 5], [20, 30, 10]]
FALLBACK_RF_Y = [0, 0, 0, 1, 1, 1]
CSV_FILENAME = "Original_full_data (1)_2.csv"


def _build_rf_training_data(csv_bytes: bytes | None) -> tuple[list[list[int]], list[int], str, int]:
    """Build training rows grouped by website; use a labelled fallback when needed."""
    dataframe: pd.DataFrame | None = None
    source = "Synthetic baseline examples"
    local_path = Path(__file__).resolve().parent / CSV_FILENAME

    try:
        if csv_bytes:
            dataframe = pd.read_csv(BytesIO(csv_bytes))
            source = "Uploaded CSV"
        elif local_path.exists():
            dataframe = pd.read_csv(local_path)
            source = f"Local file: {CSV_FILENAME}"
    except Exception:
        dataframe = None
        source = "Synthetic baseline examples (CSV could not be read)"

    if dataframe is not None:
        expected = {"webURL", "Description", "initialImpactScore"}
        if expected.issubset(set(dataframe.columns)):
            by_url: dict[str, dict[str, int]] = {}
            for _, row in dataframe.iterrows():
                url = str(row.get("webURL", "")).strip()
                if not url or url.lower() == "nan":
                    continue
                desc_value = row.get("Description", "")
                desc = "" if pd.isna(desc_value) else str(desc_value).lower()
                impact_value = pd.to_numeric(row.get("initialImpactScore", 0), errors="coerce")
                impact = int(impact_value) if pd.notna(impact_value) else 0
                stats = by_url.setdefault(
                    url,
                    {"missing_alts": 0, "empty_links": 0, "unlabeled_buttons": 0, "max_impact": 0},
                )
                stats["max_impact"] = max(stats["max_impact"], impact)
                if "alternate text" in desc:
                    stats["missing_alts"] += 1
                if "links have discernible text" in desc:
                    stats["empty_links"] += 1
                if "buttons have discernible text" in desc:
                    stats["unlabeled_buttons"] += 1

            x_rows = [
                [s["missing_alts"], s["empty_links"], s["unlabeled_buttons"]]
                for s in by_url.values()
            ]
            y_rows = [1 if s["max_impact"] >= 5 else 0 for s in by_url.values()]
            # RandomForestClassifier needs at least two target classes for risk probability.
            if len(x_rows) >= 4 and len(set(y_rows)) >= 2:
                return x_rows, y_rows, source, len(by_url)
            source += " · fallback used (not enough labelled variety)"
        elif csv_bytes:
            source = "Synthetic baseline examples (CSV is missing expected columns)"

    return FALLBACK_RF_X, FALLBACK_RF_Y, source, 0


@st.cache_resource(show_spinner=False)
def train_models(csv_bytes: bytes | None = None) -> tuple[Any, Any, Any, str, int]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_nlp = vectorizer.fit_transform(SEED_TEXTS)
    nlp_model = LogisticRegression(max_iter=1000, random_state=42).fit(x_nlp, SEED_LABELS)
    rf_x, rf_y, source, website_count = _build_rf_training_data(csv_bytes)
    rf_model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced").fit(rf_x, rf_y)
    return vectorizer, nlp_model, rf_model, source, website_count


def validate_target_url(url: str) -> str:
    """Validate an HTTP(S) URL and reject private/non-public destinations."""
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete URL starting with http:// or https://.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded usernames or passwords are not supported.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("Local/private hostnames are blocked. Use a public website you are authorized to audit.")

    try:
        resolved = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve target hostname: {host}") from exc
        resolved = list({ipaddress.ip_address(info[4][0]) for info in infos})

    if not resolved or any(not address.is_global for address in resolved):
        raise ValueError("Private, loopback, and non-public IP destinations are blocked.")
    return normalized


def fetch_public_page(url: str, timeout: int) -> requests.Response:
    """Fetch a page while validating each redirect target."""
    current_url = validate_target_url(url)
    headers = {"User-Agent": "AccessifyAccessibilityAuditor/1.0 (+website audit; respectful request)"}
    redirect_codes = {301, 302, 303, 307, 308}
    for hop in range(6):
        response = requests.get(
            current_url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        if response.status_code in redirect_codes and response.headers.get("Location"):
            destination = urljoin(current_url, response.headers["Location"])
            response.close()
            if hop == 5:
                raise requests.TooManyRedirects("Too many redirects while fetching the target.")
            current_url = validate_target_url(destination)
            continue
        response.raise_for_status()
        return response
    raise requests.TooManyRedirects("Too many redirects while fetching the target.")


def _accessible_name(element: Any) -> str:
    pieces = [element.get_text(" ", strip=True)]
    for attr in ("aria-label", "title", "alt", "value"):
        value = element.get(attr)
        if value:
            pieces.append(str(value).strip())
    # Image-only links/buttons can have a name via a descendant image's alt attribute.
    pieces.extend(img.get("alt", "").strip() for img in element.find_all("img") if img.get("alt", "").strip())
    return " ".join(piece for piece in pieces if piece).strip()


def _risk_label(probability: float) -> str:
    if probability >= 70:
        return "HIGHER REVIEW PRIORITY"
    if probability >= 40:
        return "MODERATE REVIEW PRIORITY"
    return "LOWER REVIEW PRIORITY"


def audit_website(
    url: str,
    vectorizer: Any,
    nlp_model: Any,
    rf_model: Any,
    timeout: int,
) -> dict[str, Any]:
    safe_url = validate_target_url(url)
    started = time.perf_counter()
    response = fetch_public_page(safe_url, timeout)
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and response.text.lstrip()[:100].lower().find("<html") == -1:
        raise ValueError(f"The target did not return an HTML page (content type: {content_type or 'unknown'}).")

    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.find_all("img")
    missing_alt = sum(1 for image in images if not image.has_attr("alt"))
    empty_alt = sum(1 for image in images if image.has_attr("alt") and not str(image.get("alt", "")).strip())
    all_alt_texts = [str(image.get("alt", "")) for image in images]

    links = soup.find_all("a", href=True)
    empty_links = sum(1 for link in links if not _accessible_name(link))

    buttons = soup.find_all("button")
    input_buttons = soup.find_all("input", attrs={"type": ["button", "submit", "reset", "image"]})
    unlabeled_buttons = sum(1 for button in buttons if not _accessible_name(button))
    unlabeled_buttons += sum(1 for button in input_buttons if not _accessible_name(button))

    semantic_rows: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        alt = str(image.get("alt", ""))
        if not image.has_attr("alt"):
            semantic_rows.append({"Image #": index, "Alt text": "", "Finding": "Missing alt attribute", "Confidence": "—"})
            continue
        if not alt.strip():
            semantic_rows.append({"Image #": index, "Alt text": "", "Finding": "Empty alt — verify decorative", "Confidence": "—"})
            continue
        transformed = vectorizer.transform([alt])
        prediction = int(nlp_model.predict(transformed)[0])
        confidence = float(nlp_model.predict_proba(transformed)[0][prediction]) * 100
        finding = "Likely descriptive" if prediction == 1 else "Review: possibly generic"
        semantic_rows.append({"Image #": index, "Alt text": alt, "Finding": finding, "Confidence": f"{confidence:.0f}%"})

    features = [[missing_alt, empty_links, unlabeled_buttons]]
    predicted_risk = int(rf_model.predict(features)[0])
    classes = list(rf_model.classes_)
    risk_probability = 100 * float(rf_model.predict_proba(features)[0][classes.index(1)]) if 1 in classes else 0.0
    elapsed = time.perf_counter() - started
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": safe_url,
        "final_url": response.url,
        "status_code": int(response.status_code),
        "page_title": title,
        "elapsed_seconds": round(elapsed, 2),
        "images_total": len(images),
        "missing_alt": int(missing_alt),
        "empty_alt": int(empty_alt),
        "links_total": len(links),
        "empty_links": int(empty_links),
        "buttons_total": len(buttons) + len(input_buttons),
        "unlabeled_buttons": int(unlabeled_buttons),
        "risk_prediction": predicted_risk,
        "risk_probability": round(risk_probability, 1),
        "risk_label": _risk_label(risk_probability),
        "feature_importances": {
            "Missing alt attributes": float(rf_model.feature_importances_[0]),
            "Links without accessible names": float(rf_model.feature_importances_[1]),
            "Unlabeled buttons": float(rf_model.feature_importances_[2]),
        },
        "semantic_rows": semantic_rows,
        "html_bytes": len(response.content),
        "all_alt_texts": all_alt_texts,
    }


def add_history(result: dict[str, Any]) -> None:
    st.session_state.ax_runs.insert(
        0,
        {
            "Time": result["timestamp"],
            "Target": result["target"],
            "HTTP": result["status_code"],
            "Images": result["images_total"],
            "Missing alt": result["missing_alt"],
            "Empty links": result["empty_links"],
            "Unlabeled buttons": result["unlabeled_buttons"],
            "Risk estimate": result["risk_probability"],
            "Duration (s)": result["elapsed_seconds"],
        },
    )
    st.session_state.ax_last_audit = result


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="ax-section"><div class="ax-section-title">{title}</div><div class="ax-section-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ax-hero"><span class="ax-pill"><span class="ax-dot"></span>{kicker}</span>'
        f'<div class="ax-title">{title}</div><p class="ax-subtitle">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_audit_results(result: dict[str, Any], rf_model: Any) -> None:
    section("Audit results", f"HTTP {result['status_code']} · {result['elapsed_seconds']:.2f}s")
    tabs = st.tabs(["📊 Overview", "🧠 Model signals", "📝 Semantic log"])
    with tabs[0]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images scanned", result["images_total"])
        m2.metric("Missing alt", result["missing_alt"], help="Images with no alt attribute at all.")
        m3.metric("Empty links", result["empty_links"])
        m4.metric("Unlabeled controls", result["unlabeled_buttons"])
        st.markdown(" ")
        risk_left, risk_right = st.columns([1, 1.3], gap="large")
        with risk_left:
            with st.container(border=True):
                st.markdown('<div class="ax-eyebrow">EXPERIMENTAL MODEL ESTIMATE</div>', unsafe_allow_html=True)
                st.markdown(f"### {result['risk_probability']:.1f}% risk probability")
                st.progress(min(100, max(0, int(round(result["risk_probability"])))), text=result["risk_label"])
                st.caption("This is a prototype Random Forest estimate, not a WCAG compliance score or certification.")
        with risk_right:
            with st.container(border=True):
                st.markdown("**Target page**")
                st.write(result["page_title"])
                st.caption(result["final_url"])
                c1, c2, c3 = st.columns(3)
                c1.metric("HTTP status", result["status_code"])
                c2.metric("Page size", f"{result['html_bytes'] / 1024:.1f} KB")
                c3.metric("Duration", f"{result['elapsed_seconds']:.2f}s")
        if result["empty_alt"]:
            st.info(f"{result['empty_alt']} image(s) have an explicitly empty alt attribute. That can be correct for decorative images; verify the context rather than automatically treating every one as an error.")
    with tabs[1]:
        st.markdown("### Random Forest feature importances")
        st.caption("Shows which input features influenced this trained model overall; it is not a page-specific explanation.")
        feature_frame = pd.DataFrame(
            {"Feature": list(result["feature_importances"].keys()), "Importance": list(result["feature_importances"].values())}
        ).set_index("Feature")
        st.bar_chart(feature_frame)
        st.markdown("### Image alt-text classifier")
        semantic = pd.DataFrame(result["semantic_rows"])
        if semantic.empty:
            st.info("No image elements were found on the page.")
        else:
            counts = semantic["Finding"].value_counts().rename_axis("Finding").reset_index(name="Images")
            st.dataframe(counts, hide_index=True, use_container_width=True)
            st.caption("Alt-text classifications use a small starter dataset and should be treated as review suggestions, not ground truth.")
    with tabs[2]:
        semantic = pd.DataFrame(result["semantic_rows"])
        if semantic.empty:
            st.info("No image elements were found on the page.")
        else:
            issue_filter = st.selectbox(
                "Filter image findings",
                ["All findings", "Issues / review", "Likely descriptive"],
                key=f"semantic_filter_{result['timestamp']}",
            )
            if issue_filter == "Issues / review":
                semantic = semantic[semantic["Finding"] != "Likely descriptive"]
            elif issue_filter == "Likely descriptive":
                semantic = semantic[semantic["Finding"] == "Likely descriptive"]
            st.dataframe(semantic, hide_index=True, use_container_width=True, height=360)
            csv_bytes = semantic.to_csv(index=False).encode("utf-8")
            st.download_button("Download semantic findings CSV", csv_bytes, "accessify_semantic_findings.csv", "text/csv")


# --- Audit execution helper ----------------------------------------------------
def _run_audit_action(url: str, timeout: int, vectorizer: Any, nlp_model: Any, rf_model: Any) -> None:
    url = url.strip()
    if not url:
        st.error("Enter a target URL first.")
        return
    # Avoid leaving an older result visible if this scan fails.
    st.session_state.ax_last_audit = None
    terminal = st.empty() if st.session_state.ax_settings.get("show_terminal", True) else None
    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(f"> {message}")
        if terminal is not None:
            terminal.code("\n".join(logs), language="bash")

    try:
        log("INITIALIZING ACCESSIFY PROTOCOL")
        log(f"VALIDATING TARGET: {url}")
        with st.spinner("Fetching page and running accessibility checks…"):
            result = audit_website(url, vectorizer, nlp_model, rf_model, timeout)
        log(f"HTTP RESPONSE RECEIVED: {result['status_code']}")
        log(f"DOM PARSED: {result['images_total']} IMAGE NODES LOCATED")
        log("RUNNING ALT-TEXT CLASSIFIER AND STRUCTURAL COUNTS")
        log("AUDIT COMPLETE — RENDERING RESULTS")
        st.session_state.ax_audit_logs = logs
        add_history(result)
        if terminal is not None:
            terminal.empty()
        st.toast("ACCESSIFY AUDIT COMPLETE", icon="⚡")
        st.success(f"Audit complete for {result['final_url']}")
    except (requests.RequestException, ValueError, socket.gaierror) as exc:
        if terminal is not None:
            terminal.empty()
        st.error(f"Audit could not be completed: {exc}")
    except Exception as exc:
        if terminal is not None:
            terminal.empty()
        st.error(f"Unexpected audit error: {exc}")


# --- Sidebar navigation + optional training data --------------------------------
with st.sidebar:
    st.markdown(
        '<div class="ax-brand"><div class="ax-mark">A</div><div><div class="ax-brandname">accessify<span style="color:#6be9dd">.</span></div><div class="ax-brandline">ACCESSIBILITY INTELLIGENCE</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ax-eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Website audit", "Alt-text lab", "Models & data", "Run history", "Settings"],
        label_visibility="collapsed",
        key="ax_navigation",
    )
    st.markdown("---")
    st.markdown('<div class="ax-eyebrow">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin:10px 0 8px"><span class="ax-status"><span class="ax-dot ax-live"></span> AUDIT ENGINE READY</span></div>'
        '<div style="font-size:11px;color:#8492ad;line-height:1.8">HTTP scanner · NLP prototype<br>Session-based results</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="ax-eyebrow">MODEL DATA</div>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader(
        "Optional training CSV",
        type=["csv"],
        help="Expected columns: webURL, Description, initialImpactScore. Uploading this replaces the local CSV for this session.",
        key="ax_training_upload",
    )
    st.caption("If not uploaded, Accessify looks for its training CSV beside app.py, then uses a small fallback dataset.")
    st.markdown('<div class="ax-mini">ACCESSIFY MK1 · PROTOTYPE</div>', unsafe_allow_html=True)

csv_bytes = uploaded_csv.getvalue() if uploaded_csv is not None else None
vectorizer, nlp_model, rf_model, model_source, training_websites = train_models(csv_bytes)


# --- Overview -----------------------------------------------------------------
if page == "Overview":
    hero(
        "AUTOMATED ACCESSIBILITY AUDITOR",
        'Make the web more <span class="ax-gradient">accessible.</span>',
        "Scan a public webpage for missing image alt attributes, links without accessible names, and unlabeled buttons — then review an experimental model-assisted semantic audit.",
    )
    total_runs = len(st.session_state.ax_runs)
    last = st.session_state.ax_last_audit
    total_missing = sum(int(run.get("Missing alt", 0)) for run in st.session_state.ax_runs)
    total_flags = sum(int(run.get("Empty links", 0)) + int(run.get("Unlabeled buttons", 0)) for run in st.session_state.ax_runs)
    a, b, c, d = st.columns(4)
    a.metric("AUDITS THIS SESSION", total_runs)
    b.metric("MISSING ALT FLAGS", total_missing)
    c.metric("LINK / CONTROL FLAGS", total_flags)
    d.metric("LAST AUDIT", f"{last['elapsed_seconds']:.2f}s" if last else "—")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        section("Launch an audit", "PUBLIC WEBSITE URL")
        with st.container(border=True):
            quick_url = st.text_input("Target URL", value="https://en.wikipedia.org/wiki/Main_Page", key="ax_quick_url")
            q1, q2 = st.columns([1, 1])
            with q1:
                quick_timeout = st.slider("Request timeout", 5, 30, int(st.session_state.ax_settings["timeout"]), key="ax_quick_timeout")
            with q2:
                st.markdown(" ")
                st.caption("Only scan websites you own or have permission to test.")
            if st.button("⚡ RUN ACCESSIBILITY AUDIT", type="primary", use_container_width=True, key="ax_quick_audit"):
                _run_audit_action(quick_url, quick_timeout, vectorizer, nlp_model, rf_model)
    with right:
        section("Latest result", "LIVE SESSION SNAPSHOT")
        with st.container(border=True):
            if not last:
                st.markdown('<div class="ax-mini">AWAITING FIRST AUDIT</div>', unsafe_allow_html=True)
                st.markdown("### Your dashboard is ready")
                st.caption("Run your first scan to populate metrics, classifier findings, and model diagnostics.")
                st.progress(0, text="No audit has run yet")
            else:
                st.markdown(f'<span class="ax-status">HTTP {last["status_code"]} · {last["elapsed_seconds"]:.2f}s</span>', unsafe_allow_html=True)
                st.markdown("### " + str(last["page_title"]))
                st.caption(last["final_url"])
                r1, r2, r3 = st.columns(3)
                r1.metric("Images", last["images_total"])
                r2.metric("Missing alt", last["missing_alt"])
                r3.metric("Risk estimate", f"{last['risk_probability']:.0f}%")
                st.caption("Risk estimate is experimental and is not an accessibility compliance score.")
        section("Audit pipeline", "HOW IT WORKS")
        with st.container(border=True):
            for n, title, desc in [
                ("01", "Fetch page", "Request HTML and parse the document structure."),
                ("02", "Inspect controls", "Count missing image alt attributes and unnamed links/buttons."),
                ("03", "Classify & report", "Apply prototype ML models and export the findings."),
            ]:
                st.markdown(
                    f'<div class="ax-list-row"><div style="display:flex;gap:12px;align-items:center"><span style="color:#70e9df;font:12px DM Mono,monospace">{n}</span><div><div class="ax-list-title">{title}</div><div class="ax-list-sub">{desc}</div></div></div><span class="ax-tag">PIPELINE</span></div>',
                    unsafe_allow_html=True,
                )
    if st.session_state.ax_last_audit:
        show_audit_results(st.session_state.ax_last_audit, rf_model)
    st.markdown('<div class="ax-footer"><span>ACCESSIFY / ACCESSIBILITY INTELLIGENCE</span><span>PROTOTYPE · SESSION-BASED HISTORY</span></div>', unsafe_allow_html=True)


# --- Website audit -------------------------------------------------------------
elif page == "Website audit":
    hero(
        "AUDIT CONTROL ROOM",
        'Inspect the page. <span class="ax-gradient">Find the friction.</span>',
        "Run a structural scan and inspect each image's alt-text classification. Accessify fetches static HTML; content rendered only by JavaScript may not be visible to this scanner.",
    )
    with st.form("ax_audit_form", clear_on_submit=False):
        target_url = st.text_input("Target website URL", value="https://en.wikipedia.org/wiki/Main_Page", placeholder="https://yourwebsite.com")
        c1, c2 = st.columns(2)
        with c1:
            timeout = st.slider("Request timeout (seconds)", 5, 30, int(st.session_state.ax_settings["timeout"]))
        with c2:
            st.markdown(" ")
            st.caption("HTTP(S) public URLs only · request timeout applies to the initial page request")
        start_audit = st.form_submit_button("⚡ EXECUTE AUDIT SEQUENCE", type="primary", use_container_width=True)
    if start_audit:
        _run_audit_action(target_url, timeout, vectorizer, nlp_model, rf_model)
    result = st.session_state.ax_last_audit
    if result:
        show_audit_results(result, rf_model)


# --- Alt-text lab --------------------------------------------------------------
elif page == "Alt-text lab":
    hero(
        "MANUAL NLP INJECTION",
        'Test the words behind <span class="ax-gradient">the image.</span>',
        "Check a single alt-text candidate against Accessify's starter NLP classifier. Treat the result as a suggestion for human review, not an automatic accessibility verdict.",
    )
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        section("Alt-text workbench", "SINGLE STRING CLASSIFICATION")
        with st.container(border=True):
            alt_input = st.text_area("Alt text candidate", placeholder="e.g. A golden retriever catching a red frisbee", height=110, key="ax_lab_input")
            run_lab = st.button("🧠 ANALYZE ALT TEXT", type="primary", use_container_width=True)
            if run_lab:
                value = alt_input.strip()
                if not value:
                    st.error("This string is empty. If it describes an informative image, add a concise description; if the image is purely decorative, an empty alt attribute may be appropriate.")
                else:
                    pred = int(nlp_model.predict(vectorizer.transform([value]))[0])
                    confidence = float(nlp_model.predict_proba(vectorizer.transform([value]))[0][pred]) * 100
                    if pred == 1:
                        st.success(f"Prototype result: likely descriptive · model confidence {confidence:.1f}%")
                    else:
                        st.warning(f"Prototype result: review for generic or unhelpful wording · model confidence {confidence:.1f}%")
                    st.caption("The model was trained on a small set of starter examples. Confidence is not a real-world accuracy guarantee.")
    with right:
        section("Examples", "TRY THESE")
        with st.container(border=True):
            examples = [
                ("Likely too generic", "image.png"),
                ("Likely descriptive", "A bar chart showing quarterly sales growth"),
                ("Depends on context", "Accessify logo"),
                ("Likely too generic", "photo"),
            ]
            for label, example in examples:
                st.markdown(f'<div class="ax-list-row"><div><div class="ax-list-title">{label}</div><div class="ax-list-sub">{example}</div></div><span class="ax-tag">EXAMPLE</span></div>', unsafe_allow_html=True)
            st.markdown(" ")
            st.markdown("<div class='ax-callout'><b>Good alt text is context-dependent.</b><br>Describe the image's relevant information in the context of the page. Decorative images often should use an empty alt attribute.</div>", unsafe_allow_html=True)


# --- Models & data -------------------------------------------------------------
elif page == "Models & data":
    hero(
        "MODEL DIAGNOSTICS",
        'Know what powers <span class="ax-gradient">the audit.</span>',
        "Inspect the training-data source, feature signals, and model limitations. Upload your project CSV in the sidebar to train the Random Forest using your dataset.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("NLP training examples", len(SEED_TEXTS))
    c2.metric("CSV websites used", training_websites if training_websites else "Fallback")
    c3.metric("Risk model trees", rf_model.n_estimators)
    section("Training source", "RANDOM FOREST")
    with st.container(border=True):
        st.write(model_source)
        st.caption("Expected CSV columns: webURL, Description, initialImpactScore. The model groups rows by URL and labels websites as higher-impact when the maximum impact score is at least 5.")
        st.markdown('<div class="ax-callout"><b>Prototype limitation:</b> if the CSV is missing, too small, invalid, or has only one label class, a synthetic baseline dataset is used. This risk estimate is not calibrated and must not be presented as a formal WCAG score.</div>', unsafe_allow_html=True)
    section("Feature importance", "GLOBAL MODEL SIGNALS")
    importances = pd.DataFrame(
        {"Feature": ["Missing alt attributes", "Links without accessible names", "Unlabeled buttons"], "Importance": rf_model.feature_importances_}
    ).set_index("Feature")
    st.bar_chart(importances)
    section("Model details", "CURRENT BUILD")
    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            st.markdown("**Alt-text classifier**")
            st.write("TF-IDF text vectors + Logistic Regression")
            st.caption(f"{len(SEED_TEXTS)} starter examples · binary prototype labels")
    with d2:
        with st.container(border=True):
            st.markdown("**Overall risk estimator**")
            st.write("Random Forest classifier")
            st.caption("Three structural counts · dataset or synthetic fallback")


# --- Run history ---------------------------------------------------------------
elif page == "Run history":
    hero(
        "SESSION AUDIT TRAIL",
        'Every scan. <span class="ax-gradient">One timeline.</span>',
        "Search recent audit runs, compare structural findings, and export the history for your project report or demo.",
    )
    history = pd.DataFrame(st.session_state.ax_runs)
    if history.empty:
        st.info("No audits recorded yet. Run a website audit to populate this page.")
    else:
        search = st.text_input("Filter by target URL", placeholder="Search domain or path")
        if search.strip():
            history = history[history["Target"].str.contains(search.strip(), case=False, na=False, regex=False)]
        st.dataframe(history, hide_index=True, use_container_width=True, height=420)
        st.download_button(
            "⬇ Export audit history CSV",
            history.to_csv(index=False).encode("utf-8"),
            file_name="accessify_audit_history.csv",
            mime="text/csv",
            type="primary",
        )
        if st.button("Clear session history", help="Only removes history from this Streamlit session."):
            st.session_state.ax_runs = []
            st.session_state.ax_last_audit = None
            st.rerun()
    st.caption("History lives in Streamlit session memory. Add a database if you need records to survive restarts or be shared between users.")


# --- Settings ------------------------------------------------------------------
elif page == "Settings":
    hero(
        "SYSTEM PREFERENCES",
        'Tune your <span class="ax-gradient">audit workspace.</span>',
        "Set the default timeout and terminal display preference for future scans.",
    )
    with st.form("ax_settings_form"):
        new_timeout = st.slider("Default request timeout (seconds)", 5, 30, int(st.session_state.ax_settings["timeout"]))
        show_terminal = st.toggle("Show execution log while auditing", value=bool(st.session_state.ax_settings["show_terminal"]))
        save_settings = st.form_submit_button("Save preferences", type="primary")
    if save_settings:
        st.session_state.ax_settings = {"timeout": new_timeout, "show_terminal": show_terminal}
        st.success("Preferences saved for this session.")
    section("Privacy & limitations", "READ BEFORE DEPLOYMENT")
    with st.container(border=True):
        st.markdown("**Requests:** Accessify fetches the URL you enter. Audit only sites you own or are authorized to test.")
        st.markdown("**JavaScript:** This version inspects the HTML response and does not run client-side JavaScript.")
        st.markdown("**Accessibility:** Counts are automated heuristics. They do not replace keyboard testing, screen-reader review, or a full WCAG audit.")
        st.markdown("**Storage:** Results and uploaded CSV data are held in the active Streamlit session; no database is configured.")
