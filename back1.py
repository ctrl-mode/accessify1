from __future__ import annotations

import base64
import html
import ipaddress
import json
import os
import re
import socket
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
      .block-container { padding-top:1.25rem; padding-bottom:3rem; max-width:none !important; width:100% !important; padding-left:clamp(1rem,2.5vw,3rem); padding-right:clamp(1rem,2.5vw,3rem); }
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
      .ax-chart-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:2px 0 7px; }
      .ax-chart-kicker { color:#8b9ab8; font:10px 'DM Mono',monospace; letter-spacing:1.45px; text-transform:uppercase; }
      .ax-chart-live { color:#8bf3df; background:rgba(105,240,228,.07); border:1px solid rgba(105,240,228,.18); padding:5px 8px; border-radius:7px; font:9px 'DM Mono',monospace; letter-spacing:.5px; white-space:nowrap; }
      .ax-signal-card { min-height:300px; padding:18px; border-radius:15px; border:1px solid rgba(139,158,204,.15); background:radial-gradient(circle at 100% 0%,rgba(105,240,228,.09),transparent 35%),linear-gradient(145deg,rgba(20,29,46,.8),rgba(11,16,28,.74)); }
      .ax-signal-top { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:19px; }
      .ax-signal-index { color:#7888a8; font:10px 'DM Mono',monospace; letter-spacing:1.2px; }
      .ax-signal-badge { color:#a1f4e7; font:9px 'DM Mono',monospace; letter-spacing:.7px; border:1px solid rgba(105,240,228,.2); background:rgba(105,240,228,.07); border-radius:6px; padding:5px 7px; }
      .ax-signal-name { color:#dfe8fb; font-size:14px; font-weight:750; line-height:1.5; margin-bottom:6px; }
      .ax-signal-value { color:#73f0e0; font-size:42px; font-weight:850; letter-spacing:-2px; line-height:1.1; margin:7px 0 3px; }
      .ax-signal-caption { color:#8797b3; font-size:10px; line-height:1.6; }
      .ax-signal-track { height:6px; background:rgba(144,162,200,.13); border-radius:20px; overflow:hidden; margin:14px 0 17px; }
      .ax-signal-track > span { display:block; height:100%; border-radius:20px; background:linear-gradient(90deg,#7e8cff,#69f0e4); box-shadow:0 0 12px rgba(105,240,228,.2); }
      .ax-signal-divider { height:1px; background:rgba(145,163,200,.12); margin:14px 0; }
      .ax-rank-row { display:flex; align-items:center; gap:9px; padding:8px 0; border-bottom:1px solid rgba(145,163,200,.08); }
      .ax-rank-row:last-child { border-bottom:0; }
      .ax-rank-num { width:22px; height:22px; flex:0 0 22px; display:grid; place-items:center; border-radius:7px; color:#99a8c5; background:rgba(145,163,200,.08); font:10px 'DM Mono',monospace; }
      .ax-rank-name { flex:1; color:#aab7d0; font-size:10px; line-height:1.45; }
      .ax-rank-value { color:#e5ecfb; font:11px 'DM Mono',monospace; white-space:nowrap; }
      .ax-chart-foot { color:#75839f; font-size:10px; line-height:1.65; margin-top:5px; }
      .ax-footer { margin-top:35px; padding-top:14px; border-top:1px solid rgba(145,163,200,.1); display:flex; justify-content:space-between; gap:10px; color:#65728c; font:9px 'DM Mono',monospace; letter-spacing:1px; }
      @keyframes axPulse { 0%,100%{opacity:.55} 50%{opacity:1} } .ax-live { animation:axPulse 2s ease-in-out infinite; }
      .ax-feature-card { border:1px solid rgba(145,163,200,.15); background:linear-gradient(145deg,rgba(20,28,44,.85),rgba(10,15,26,.9)); padding:16px 18px; border-radius:15px; min-height:118px; }
      .ax-feature-card h4 { margin:0 0 7px 0; font-size:13px; letter-spacing:0; }
      .ax-feature-card p { margin:0; color:#91a0bc; font-size:11px; line-height:1.65; }
      .ax-code-sample { border:1px solid rgba(105,240,228,.16); border-radius:12px; background:rgba(5,10,18,.6); padding:10px 12px; }
      .ax-full-width-note { color:#8a99b6; font:10px 'DM Mono',monospace; letter-spacing:.7px; }
      /* Interaction polish: visible keyboard focus, calm hover states, responsive navigation. */
      :focus-visible { outline:2px solid #69f0e4 !important; outline-offset:3px !important; }
      .stButton>button:focus-visible,.stDownloadButton>button:focus-visible { box-shadow:0 0 0 3px rgba(105,240,228,.18) !important; }
      [data-testid="stSidebar"] [role="radiogroup"] { gap:5px; }
      [data-testid="stSidebar"] [role="radiogroup"] label { border:1px solid transparent; border-radius:10px; padding:7px 10px; transition:background .16s ease,border-color .16s ease; }
      [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(105,240,228,.055); border-color:rgba(105,240,228,.12); }
      [data-testid="stVerticalBlockBorderWrapper"] { transition:border-color .18s ease, transform .18s ease; }
      [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:rgba(105,240,228,.23)!important; }
      .stTextInput input:focus,.stTextArea textarea:focus { border-color:rgba(105,240,228,.65)!important; box-shadow:0 0 0 2px rgba(105,240,228,.08)!important; }
      [data-testid="stDataFrame"] { box-shadow:0 10px 28px rgba(0,0,0,.10); }
      @media(max-width:700px) { .ax-hero{padding:22px 19px} .ax-title{letter-spacing:-1px} .block-container {padding-left:.75rem;padding-right:.75rem;} }
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
CSV_FILENAME = "Original_full_data (1).csv"


def _build_rf_training_data(csv_bytes: bytes | None) -> tuple[list[list[int]], list[int], str, int, int]:
    """Build training rows grouped by website; use a labelled fallback when needed."""
    dataframe: pd.DataFrame | None = None
    source = "Synthetic baseline examples"
    local_path = Path(__file__).resolve().parent / CSV_FILENAME

    try:
        if csv_bytes:
            dataframe = pd.read_csv(BytesIO(csv_bytes), low_memory=False)
            source = f"Uploaded CSV · {len(dataframe):,} records"
        elif local_path.exists():
            dataframe = pd.read_csv(local_path, low_memory=False)
            source = f"Bundled dataset: {CSV_FILENAME} · {len(dataframe):,} records"
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
                return x_rows, y_rows, source, len(by_url), len(dataframe)
            source += " · fallback used (not enough labelled variety)"
        elif csv_bytes:
            source = f"Synthetic baseline examples (CSV is missing expected columns; {len(dataframe):,} rows were not used)"

    return FALLBACK_RF_X, FALLBACK_RF_Y, source, 0, len(dataframe) if dataframe is not None else 0


@st.cache_resource(show_spinner=False)
def train_models(csv_bytes: bytes | None = None) -> tuple[Any, Any, Any, str, int, int]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_nlp = vectorizer.fit_transform(SEED_TEXTS)
    nlp_model = LogisticRegression(max_iter=1000, random_state=42).fit(x_nlp, SEED_LABELS)
    rf_x, rf_y, source, website_count, record_count = _build_rf_training_data(csv_bytes)
    rf_model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced").fit(rf_x, rf_y)
    return vectorizer, nlp_model, rf_model, source, website_count, record_count


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


def _element_line(element: Any) -> int | str:
    line = getattr(element, "sourceline", None)
    return int(line) if line is not None else "Unavailable"


def _element_snippet(element: Any, limit: int = 650) -> str:
    try:
        snippet = str(element).replace("\n", " ").strip()
    except Exception:
        snippet = ""
    return snippet[:limit] + ("…" if len(snippet) > limit else "")


def _fix_for_issue(issue_key: str, element: Any = None) -> str:
    """Return an explanatory, reviewable fix example. Suggestions are not auto-applied."""
    if issue_key == "missing_alt":
        src = str(element.get("src", "image.png")) if element else "image.png"
        return f'<img src="{src}" alt="Concise description of the informative image">'
    if issue_key == "empty_alt":
        return 'If decorative: <img src="decorative.svg" alt="">\nIf informative: use concise text describing its purpose in this page context.'
    if issue_key == "generic_alt":
        return '<!-- Replace generic wording with context-specific information -->\n<img src="chart.png" alt="Bar chart comparing quarterly sales from Q1 to Q4">'
    if issue_key == "unnamed_link":
        return '<a href="/report" aria-label="Open the accessibility report">\n  <svg aria-hidden="true">…</svg>\n</a>\n<!-- Prefer visible link text when possible. -->'
    if issue_key == "generic_link":
        return '<a href="/pricing">View pricing plans</a>\n<!-- Replace vague text such as “click here” or “read more”. -->'
    if issue_key == "unnamed_button":
        return '<button type="button" aria-label="Open settings">\n  <svg aria-hidden="true">…</svg>\n</button>\n<!-- Or provide visible button text such as “Open settings”. -->'
    if issue_key == "unlabeled_control":
        return '<label for="email">Email address</label>\n<input id="email" name="email" type="email">'
    if issue_key == "missing_title":
        return '<head>\n  <title>Page topic | Site name</title>\n</head>'
    if issue_key == "missing_lang":
        return '<html lang="en">\n  <!-- Set the correct BCP 47 language tag for the page. -->'
    if issue_key == "heading_jump":
        return '<h2>Section heading</h2>\n<h3>Subsection heading</h3>\n<!-- Use headings to reflect hierarchy; do not skip levels merely for styling. -->'
    if issue_key == "iframe_title":
        return '<iframe src="/map" title="Map showing the service location"></iframe>'
    if issue_key == "duplicate_id":
        return '<!-- Give each id a unique value; update matching label/ARIA references too. -->'
    if issue_key == "empty_summary":
        return '<summary>Delivery options</summary>\n<!-- The summary should communicate what the disclosure contains. -->'
    return "Review the element's accessible name, semantics, and purpose in context. Confirm with keyboard and screen-reader testing."


def inspect_accessibility(soup: BeautifulSoup, vectorizer: Any, nlp_model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Run deterministic HTML checks and add line-level image text review signals."""
    findings: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    nlp_logs: list[str] = []

    def add(issue_key: str, title: str, severity: str, element: Any, detail: str, category: str = "Structure") -> None:
        findings.append({
            "Severity": severity,
            "Category": category,
            "Issue": title,
            "Source line": _element_line(element) if element is not None else "Unavailable",
            "Element": getattr(element, "name", "document") if element is not None else "document",
            "Details": detail,
            "HTML evidence": _element_snippet(element) if element is not None else "",
            "Suggested fix": _fix_for_issue(issue_key, element),
        })

    images = soup.find_all("img")
    for index, image in enumerate(images, start=1):
        alt = str(image.get("alt", "")) if image.has_attr("alt") else ""
        line = _element_line(image)
        line_label = str(line) if line != "Unavailable" else "unknown"
        if not image.has_attr("alt"):
            log_entry = f"🚫 Missing alt attribute in {line_label} line"
            nlp_logs.append(log_entry)
            semantic_rows.append({"Image #": index, "Source line": line, "Alt text": "<missing alt attribute>", "Finding": "Missing alt attribute", "Confidence": "—", "NLP log": log_entry})
            add("missing_alt", "Image missing alt attribute", "High", image, "Informative images need an appropriate text alternative. Decorative images may use alt=\"\".", "Images")
            continue
        if not alt.strip():
            log_entry = f"🚫 Lazy code in {line_label} line: {json.dumps(alt, ensure_ascii=False)}"
            nlp_logs.append(log_entry)
            semantic_rows.append({"Image #": index, "Source line": line, "Alt text": alt, "Finding": "Empty alt — verify decorative", "Confidence": "—", "NLP log": log_entry})
            add("empty_alt", "Empty alt text — review purpose", "Review", image, "An empty alt attribute is correct for a decorative image, but not usually for an informative image.", "Images")
            continue
        transformed = vectorizer.transform([alt])
        prediction = int(nlp_model.predict(transformed)[0])
        confidence = float(nlp_model.predict_proba(transformed)[0][prediction]) * 100
        quoted = json.dumps(alt, ensure_ascii=False)
        if prediction == 0:
            log_entry = f"🚫 AI Flagged Junk in {line_label} line: {quoted}"
            nlp_logs.append(log_entry)
            finding = "Review: possibly generic"
            add("generic_alt", "Alt text may be generic", "Medium", image, f"The starter NLP model flagged {quoted}. Confirm whether it conveys the image's purpose in context.", "Images / NLP review")
        else:
            log_entry = f"✅ Alt text looks descriptive in {line_label} line: {quoted}"
            finding = "Likely descriptive"
        semantic_rows.append({"Image #": index, "Source line": line, "Alt text": alt, "Finding": finding, "Confidence": f"{confidence:.0f}%", "NLP log": log_entry})

    # Links without a usable accessible name, and vague link text that merits review.
    vague_link_names = {"click here", "here", "read more", "more", "learn more", "click", "link"}
    for link in soup.find_all("a", href=True):
        name = _accessible_name(link).strip()
        if not name:
            add("unnamed_link", "Link has no accessible name", "High", link, "Provide visible link text or an accessible name that describes its purpose.", "Links")
        elif name.casefold() in vague_link_names:
            add("generic_link", "Link text may be vague", "Review", link, f"The link name “{name}” may be unclear out of context.", "Links")

    # Buttons and input type buttons.
    control_buttons = list(soup.find_all("button")) + list(soup.find_all("input", attrs={"type": ["button", "submit", "reset", "image"]}))
    for button in control_buttons:
        if not _accessible_name(button):
            add("unnamed_button", "Button has no accessible name", "High", button, "Icon-only buttons need an accessible name describing the action.", "Controls")

    # Labelable form controls. Account for explicit labels, wrapping labels, and aria-label/aria-labelledby.
    label_ids = {str(label.get("for")) for label in soup.find_all("label") if label.get("for")}
    all_ids = {str(node.get("id")) for node in soup.find_all(attrs={"id": True})}
    labelable = soup.find_all(["input", "select", "textarea"])
    skip_types = {"hidden", "submit", "reset", "button", "image"}
    for control in labelable:
        typ = str(control.get("type", "text")).lower() if control.name == "input" else ""
        if typ in skip_types:
            continue
        has_label = bool(control.get("aria-label") or control.get("aria-labelledby") or control.get("title"))
        if control.get("id") and str(control.get("id")) in label_ids:
            has_label = True
        if control.find_parent("label") is not None:
            has_label = True
        if not has_label:
            add("unlabeled_control", "Form control has no associated label", "High", control, "Associate the control with a meaningful label or an appropriate accessible name.", "Forms")

    html_el = soup.find("html")
    if html_el is not None and not str(html_el.get("lang", "")).strip():
        add("missing_lang", "Document language is missing", "Medium", html_el, "Set the document language so assistive technology can use appropriate pronunciation rules.", "Document")
    title_el = soup.find("title")
    if title_el is None or not title_el.get_text(" ", strip=True):
        add("missing_title", "Document title is missing or empty", "Medium", title_el or soup.find("head") or html_el, "Provide a concise title that describes the page topic.", "Document")

    # Heading hierarchy: report jumps larger than one level, not every repeated heading.
    previous_level = None
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(heading.name[1])
        if previous_level is not None and level > previous_level + 1:
            add("heading_jump", "Heading level skips a step", "Review", heading, f"Heading level jumps from h{previous_level} to h{level}; check whether the hierarchy still makes sense.", "Headings")
        if heading.get_text(" ", strip=True):
            previous_level = level

    for frame in soup.find_all(["iframe", "frame"]):
        if not str(frame.get("title", "")).strip() and not str(frame.get("aria-label", "")).strip():
            add("iframe_title", "Embedded frame has no title", "Medium", frame, "Give each frame a concise title describing its content.", "Embedded content")

    for summary in soup.find_all("summary"):
        if not summary.get_text(" ", strip=True) and not summary.get("aria-label"):
            add("empty_summary", "Disclosure summary has no name", "High", summary, "Give the summary text that explains what will expand or collapse.", "Disclosure")

    ids: dict[str, list[Any]] = {}
    for node in soup.find_all(attrs={"id": True}):
        ids.setdefault(str(node.get("id")), []).append(node)
    for duplicate_id, nodes in ids.items():
        if duplicate_id and len(nodes) > 1:
            add("duplicate_id", "Duplicate id attribute", "Medium", nodes[1], f'The id "{duplicate_id}" appears {len(nodes)} times; IDs should be unique.', "Structure")

    return findings, semantic_rows, nlp_logs


def analyze_html_content(
    html_content: str,
    target: str,
    final_url: str,
    status_code: int,
    elapsed: float,
    vectorizer: Any,
    nlp_model: Any,
    rf_model: Any,
    html_bytes: int | None = None,
) -> dict[str, Any]:
    soup = BeautifulSoup(html_content, "html.parser")
    findings, semantic_rows, nlp_logs = inspect_accessibility(soup, vectorizer, nlp_model)
    images = soup.find_all("img")
    missing_alt = sum(1 for image in images if not image.has_attr("alt"))
    empty_alt = sum(1 for image in images if image.has_attr("alt") and not str(image.get("alt", "")).strip())
    links = soup.find_all("a", href=True)
    empty_links = sum(1 for link in links if not _accessible_name(link))
    buttons = list(soup.find_all("button")) + list(soup.find_all("input", attrs={"type": ["button", "submit", "reset", "image"]}))
    unlabeled_buttons = sum(1 for button in buttons if not _accessible_name(button))
    features = [[missing_alt, empty_links, unlabeled_buttons]]
    predicted_risk = int(rf_model.predict(features)[0])
    classes = list(rf_model.classes_)
    risk_probability = 100 * float(rf_model.predict_proba(features)[0][classes.index(1)]) if 1 in classes else 0.0
    feature_importances = {
        "Missing alt attributes": float(rf_model.feature_importances_[0]),
        "Links without accessible names": float(rf_model.feature_importances_[1]),
        "Unlabeled buttons": float(rf_model.feature_importances_[2]),
    }
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target,
        "final_url": final_url,
        "status_code": int(status_code),
        "page_title": soup.title.get_text(" ", strip=True) if soup.title else "Untitled page",
        "elapsed_seconds": round(float(elapsed), 2),
        "images_total": len(images),
        "missing_alt": int(missing_alt),
        "empty_alt": int(empty_alt),
        "links_total": len(links),
        "empty_links": int(empty_links),
        "buttons_total": len(buttons),
        "unlabeled_buttons": int(unlabeled_buttons),
        "risk_prediction": predicted_risk,
        "risk_probability": round(risk_probability, 1),
        "risk_label": _risk_label(risk_probability),
        "feature_importances": feature_importances,
        "semantic_rows": semantic_rows,
        "nlp_logs": nlp_logs,
        "audit_findings": findings,
        "html_bytes": int(html_bytes if html_bytes is not None else len(html_content.encode("utf-8", errors="replace"))),
        "all_alt_texts": [str(image.get("alt", "")) for image in images],
    }


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
    try:
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "<html" not in response.text.lstrip()[:300].lower():
            raise ValueError(f"The target did not return an HTML page (content type: {content_type or 'unknown'}).")
        return analyze_html_content(
            response.text, safe_url, response.url, response.status_code,
            time.perf_counter() - started, vectorizer, nlp_model, rf_model, len(response.content),
        )
    finally:
        response.close()


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


def render_feature_importance_chart(feature_importances: dict[str, float], key: str) -> None:
    """Render a polished, hoverable feature-importance chart with a ranked insight panel."""
    rows = [(str(name), max(0.0, float(value))) for name, value in feature_importances.items()]
    rows.sort(key=lambda item: item[1], reverse=True)
    if not rows:
        st.info("No feature importance values are available for this model.")
        return

    total = sum(value for _, value in rows)
    shares = [(name, (value / total if total else 0.0)) for name, value in rows]
    strongest_name, strongest_share = shares[0]
    strongest_escaped = html.escape(strongest_name)

    # Plotly gives the chart precise labels, hover details, and a clean horizontal layout.
    palette = ["#69F0E4", "#8492FF", "#B49BFF", "#70B7FF", "#89D9C9"]
    color_by_name = {name: palette[i % len(palette)] for i, (name, _) in enumerate(shares)}
    labels = [name for name, _ in reversed(shares)]  # Weakest at the bottom, strongest at the top.
    value_by_name = dict(shares)
    fig = go.Figure(
        go.Bar(
            x=[value_by_name[name] for name in labels],
            y=labels,
            orientation="h",
            marker={
                "color": [color_by_name[name] for name in labels],
                "line": {"width": 0},
                "opacity": 0.96,
            },
            text=[f"{value_by_name[name]:.1%}" for name in labels],
            textposition="outside",
            textfont={"family": "DM Mono, monospace", "size": 11, "color": "#dfe8fb"},
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Relative importance: %{x:.1%}<extra></extra>",
            width=0.52,
        )
    )
    max_share = max(value_by_name.values(), default=0.0)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": "#aab7d0", "size": 11},
        margin={"l": 6, "r": 38, "t": 10, "b": 8},
        height=max(245, 88 * len(rows)),
        bargap=0.44,
        showlegend=False,
        hoverlabel={"bgcolor": "#101827", "bordercolor": "#34435e", "font": {"color": "#eef3ff", "family": "Manrope"}},
        xaxis={
            "range": [0, max_share * 1.28 if max_share else 1],
            "tickformat": ".0%",
            "ticks": "",
            "showline": False,
            "showticklabels": True,
            "gridcolor": "rgba(145,163,200,.13)",
            "zeroline": False,
            "tickfont": {"color": "#7786a2", "size": 10},
            "fixedrange": True,
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": labels,
            "showgrid": False,
            "ticks": "",
            "automargin": True,
            "tickfont": {"color": "#d0d9ed", "size": 11},
            "fixedrange": True,
        },
    )

    chart_col, insight_col = st.columns([1.7, 1], gap="large")
    with chart_col:
        with st.container(border=True):
            st.markdown(
                '<div class="ax-chart-head"><div class="ax-chart-kicker">GLOBAL MODEL SIGNALS</div>'
                '<div class="ax-chart-live">● MODEL VIEW</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
                key=key,
            )
            st.markdown(
                '<div class="ax-chart-foot">Hover over a bar to inspect its relative importance. Values describe the trained model overall, not an explanation for a single page result.</div>',
                unsafe_allow_html=True,
            )

    with insight_col:
        with st.container(border=True):
            rank_html = []
            for rank, (name, share) in enumerate(shares, start=1):
                rank_html.append(
                    '<div class="ax-rank-row">'
                    f'<span class="ax-rank-num">{rank:02d}</span>'
                    f'<span class="ax-rank-name">{html.escape(name)}</span>'
                    f'<span class="ax-rank-value">{share:.1%}</span>'
                    '</div>'
                )
            insight_html = (
                f'<div class="ax-signal-card"><div class="ax-signal-top">'
                f'<span class="ax-signal-index">SIGNAL  /  01</span><span class="ax-signal-badge">TOP CONTRIBUTOR</span></div>'
                f'<div class="ax-signal-name">{strongest_escaped}</div>'
                f'<div class="ax-signal-value">{strongest_share:.1%}</div>'
                f'<div class="ax-signal-caption">of total feature importance in this trained model</div>'
                f'<div class="ax-signal-track"><span style="width:{min(100, strongest_share * 100):.2f}%"></span></div>'
                f'<div class="ax-signal-divider"></div>'
                f'<div class="ax-chart-kicker" style="margin-bottom:7px">SIGNAL RANKING</div>'
                f'{"".join(rank_html)}</div>'
            )
            st.markdown(insight_html, unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ax-hero"><span class="ax-pill"><span class="ax-dot"></span>{kicker}</span>'
        f'<div class="ax-title">{title}</div><p class="ax-subtitle">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_audit_results(result: dict[str, Any], rf_model: Any) -> None:
    section("Audit results", f"HTTP {result['status_code']} · {result['elapsed_seconds']:.2f}s")
    tabs = st.tabs(["📊 Overview", "🧠 Model signals", "📝 Semantic log", "🛠 Findings & fixes"])
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
        findings_df = pd.DataFrame(result.get("audit_findings", []))
        if not findings_df.empty:
            section("Findings distribution", "CURRENT PAGE · DETERMINISTIC CHECKS")
            chart_a, chart_b = st.columns([1, 1], gap="large")
            with chart_a:
                by_severity = findings_df.groupby("Severity", dropna=False).size().reset_index(name="Findings")
                fig = go.Figure(go.Bar(x=by_severity["Severity"], y=by_severity["Findings"], marker_color=[{"High":"#ff6b7a","Medium":"#ffbf69","Review":"#8e9eff"}.get(str(v), "#69f0e4") for v in by_severity["Severity"]], hovertemplate="%{x}<br>%{y} findings<extra></extra>"))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=10,b=5), height=260, showlegend=False, xaxis=dict(title=None), yaxis=dict(title="Finding count", gridcolor="rgba(145,163,200,.13)"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"findings_severity_{result['timestamp']}")
            with chart_b:
                by_category = findings_df.groupby("Category", dropna=False).size().sort_values(ascending=True).tail(10).reset_index(name="Findings")
                fig2 = go.Figure(go.Bar(x=by_category["Findings"], y=by_category["Category"], orientation="h", marker_color="#69f0e4", hovertemplate="%{y}<br>%{x} findings<extra></extra>"))
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=12,t=10,b=5), height=260, showlegend=False, xaxis=dict(title="Finding count", gridcolor="rgba(145,163,200,.13)"), yaxis=dict(title=None, automargin=True))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key=f"findings_category_{result['timestamp']}")
    with tabs[1]:
        st.markdown("### What drives the model?")
        st.caption("A ranked view of which structural signals influenced the trained Random Forest most.")
        render_feature_importance_chart(result["feature_importances"], key=f"audit_feature_importance_{result['timestamp']}")
        st.markdown("### Image alt-text classifier")
        semantic = pd.DataFrame(result["semantic_rows"])
        if semantic.empty:
            st.info("No image elements were found on the page.")
        else:
            counts = semantic["Finding"].value_counts().rename_axis("Finding").reset_index(name="Images")
            st.dataframe(counts, hide_index=True, use_container_width=True)
            st.caption("Alt-text classifications use a small starter dataset and should be treated as review suggestions, not ground truth.")
    with tabs[2]:
        st.markdown("### Source-line NLP log")
        st.caption("Line numbers refer to the HTML response received by Accessify. A line may be unavailable for some parser/input combinations.")
        logged_findings = result.get("nlp_logs", [])
        if logged_findings:
            st.code("\n".join(logged_findings), language="text")
        elif not result["semantic_rows"]:
            st.info("No image elements were found on the page.")
        else:
            st.success("No missing-alt, empty-alt, or prototype NLP-flagged image findings were recorded.")

        semantic = pd.DataFrame(result["semantic_rows"])
        if not semantic.empty:
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
    with tabs[3]:
        st.markdown("### Evidence-first remediation")
        st.caption("Each row contains source-line evidence and a suggested correction. Suggestions are examples; validate them against the page's context.")
        findings_df = pd.DataFrame(result.get("audit_findings", []))
        if findings_df.empty:
            st.success("No findings were returned by the implemented checks. This is not proof of full accessibility.")
        else:
            f1, f2 = st.columns(2)
            severity_options = ["All severities"] + sorted(findings_df["Severity"].dropna().astype(str).unique().tolist())
            category_options = ["All categories"] + sorted(findings_df["Category"].dropna().astype(str).unique().tolist())
            with f1:
                severity_filter = st.selectbox("Severity filter", severity_options, key=f"sev_{result['timestamp']}")
            with f2:
                category_filter = st.selectbox("Category filter", category_options, key=f"cat_{result['timestamp']}")
            filtered = findings_df.copy()
            if severity_filter != "All severities":
                filtered = filtered[filtered["Severity"] == severity_filter]
            if category_filter != "All categories":
                filtered = filtered[filtered["Category"] == category_filter]
            st.dataframe(filtered[["Severity", "Category", "Issue", "Source line", "Element", "Details"]], hide_index=True, use_container_width=True, height=320)
            chosen_index = st.selectbox("Inspect a finding", list(range(len(filtered))), format_func=lambda i: f"{filtered.iloc[i]['Severity']} · line {filtered.iloc[i]['Source line']} · {filtered.iloc[i]['Issue']}", key=f"finding_detail_{result['timestamp']}")
            selected_finding = filtered.iloc[int(chosen_index)].to_dict()
            with st.container(border=True):
                st.markdown(f"#### {selected_finding['Issue']}")
                st.write(selected_finding["Details"])
                st.caption(f"Line: {selected_finding['Source line']} · Element: {selected_finding['Element']} · Severity: {selected_finding['Severity']}")
                st.markdown("**HTML evidence**")
                st.code(str(selected_finding.get("HTML evidence", "")) or "No source snippet available", language="html")
                st.markdown("**Suggested correction**")
                st.code(str(selected_finding.get("Suggested fix", "")), language="html")
                st.download_button("Download suggested fix", str(selected_finding.get("Suggested fix", "")).encode("utf-8"), file_name="accessify_suggested_fix.html", mime="text/html", key=f"fix_download_{result['timestamp']}")


# --- Dataset analytics, document extraction, and export helpers ------------------
@st.cache_data(show_spinner=False)
def load_dataset_dataframe(csv_bytes: bytes | None = None) -> pd.DataFrame:
    local_path = Path(__file__).resolve().parent / CSV_FILENAME
    if csv_bytes:
        return pd.read_csv(BytesIO(csv_bytes), low_memory=False)
    if local_path.exists():
        return pd.read_csv(local_path, low_memory=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def evaluate_dataset_model(csv_bytes: bytes | None = None) -> dict[str, Any]:
    """Grouped holdout evaluation of impact-score labels using text/category only.

    URL groups are kept intact between train and test to reduce leakage from related
    violation records belonging to the same page.
    """
    df = load_dataset_dataframe(csv_bytes)
    required = {"webURL", "Description", "initialImpactScore"}
    if df.empty or not required.issubset(df.columns):
        return {"available": False, "reason": "Dataset is missing webURL, Description, or initialImpactScore."}
    clean = df.dropna(subset=["webURL", "initialImpactScore"]).copy()
    clean["webURL"] = clean["webURL"].astype(str)
    scores = pd.to_numeric(clean["initialImpactScore"], errors="coerce")
    clean = clean.loc[scores.notna()].copy()
    clean["label"] = (scores.loc[clean.index] >= 5).astype(int)
    if clean["label"].nunique() < 2 or clean["webURL"].nunique() < 8:
        return {"available": False, "reason": "Not enough distinct URL groups or label variety for a meaningful held-out evaluation."}
    descriptions = clean["Description"].fillna("").astype(str)
    categories = clean["category"].fillna("unknown").astype(str) if "category" in clean else pd.Series("unknown", index=clean.index)
    X_text = descriptions + " category=" + categories
    y = clean["label"].astype(int).to_numpy()
    groups = clean["webURL"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    try:
        train_idx, test_idx = next(splitter.split(X_text, y, groups=groups))
    except StopIteration:
        return {"available": False, "reason": "Could not create a URL-grouped split."}
    if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
        return {"available": False, "reason": "The URL-grouped split did not leave both labels in train and test. Try a different fixed split or more data."}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, stop_words="english", sublinear_tf=True)
    try:
        X_train = vectorizer.fit_transform(X_text.iloc[train_idx])
        X_test = vectorizer.transform(X_text.iloc[test_idx])
        model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
    except ValueError as exc:
        return {"available": False, "reason": f"Could not fit a text model on this dataset: {exc}"}
    cm = confusion_matrix(y[test_idx], pred, labels=[0, 1])
    report = classification_report(y[test_idx], pred, labels=[0, 1], target_names=["Impact score < 5", "Impact score ≥ 5"], output_dict=True, zero_division=0)
    top = sorted(zip(vectorizer.get_feature_names_out().tolist(), model.coef_[0].tolist()), key=lambda pair: abs(pair[1]), reverse=True)[:15]
    return {
        "available": True,
        "rows": len(clean),
        "urls": int(clean["webURL"].nunique()),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_urls": int(pd.Series(groups[train_idx]).nunique()),
        "test_urls": int(pd.Series(groups[test_idx]).nunique()),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "precision": float(precision_score(y[test_idx], pred, zero_division=0)),
        "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
        "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_report": report,
        "top_terms": [{"Term": term, "Model weight": weight, "Direction": "Higher impact label" if weight > 0 else "Lower impact label"} for term, weight in top],
        "label_counts": pd.Series(y).value_counts().sort_index().to_dict(),
    }


def build_report_html(result: dict[str, Any]) -> str:
    findings = result.get("audit_findings", [])
    body_rows = []
    for item in findings:
        body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(item.get(key, '')))}</td>" for key in ("Severity", "Category", "Issue", "Source line", "Details", "Suggested fix")) + "</tr>")
    title = html.escape(str(result.get("page_title", "Accessify report")))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Accessify audit report — {title}</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}}h1{{color:#087f8c}}.meta{{color:#586174}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dbe2ea;padding:10px;font-size:13px}}th{{background:#edf5f6}}code{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body><h1>Accessify audit report</h1><h2>{title}</h2><p class="meta">Target: {html.escape(str(result.get('final_url', result.get('target',''))))}<br>Generated: {html.escape(str(result.get('timestamp','')))} · HTTP {result.get('status_code','—')} · {result.get('elapsed_seconds',0)}s</p><p>Images: {result.get('images_total',0)} · Missing alt attributes: {result.get('missing_alt',0)} · Unnamed links: {result.get('empty_links',0)} · Unnamed controls: {result.get('unlabeled_buttons',0)}</p><h2>Findings and suggested fixes</h2><table><thead><tr><th>Severity</th><th>Category</th><th>Issue</th><th>Line</th><th>Details</th><th>Suggested fix</th></tr></thead><tbody>{''.join(body_rows) or '<tr><td colspan="6">No deterministic findings recorded by the selected checks.</td></tr>'}</tbody></table><p class="meta">Automated findings are heuristics, not a WCAG certification. Confirm issues with manual keyboard and screen-reader testing.</p></body></html>"""


DEMO_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Accessify demo: Inclusive campus guide</title></head>
<body>
  <h1>Inclusive campus guide</h1>
  <h4>Getting around</h4>
  <img src="campus-map.png" alt="image">
  <img src="decorative-divider.svg" alt=" ">
  <img src="campus-ramp.jpg">
  <a href="/transport"><img src="bus.svg"></a>
  <button><svg aria-hidden="true"></svg></button>
  <label>Student email <input type="email" id="student-email"></label>
  <iframe src="/map"></iframe>
  <details><summary></summary><p>Accessibility services.</p></details>
</body></html>"""


def run_demo_audit(vectorizer: Any, nlp_model: Any, rf_model: Any) -> dict[str, Any]:
    return analyze_html_content(
        DEMO_HTML, "demo://inclusive-campus-guide", "demo://inclusive-campus-guide", 200,
        0.04, vectorizer, nlp_model, rf_model, len(DEMO_HTML.encode("utf-8")),
    )


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
        ["Overview", "Website audit", "Models & data", "Remediation assistant", "Reports & demo", "Run history", "Settings"],
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
    st.caption("Accessify automatically loads the bundled Original_full_data (1).csv beside app.py. Upload another CSV only to override it for this session.")
    st.markdown('<div class="ax-mini">ACCESSIFY MK2 · ACCESSIBILITY STUDIO</div>', unsafe_allow_html=True)

csv_bytes = uploaded_csv.getvalue() if uploaded_csv is not None else None
vectorizer, nlp_model, rf_model, model_source, training_websites, training_records = train_models(csv_bytes)


# --- Overview -----------------------------------------------------------------
if page == "Overview":
    hero(
        "AUTOMATED ACCESSIBILITY AUDITOR",
        'Make the web more <span class="ax-gradient">accessible.</span>',
        "Scan a public webpage for missing alternatives, unnamed links and controls, form-label problems, document metadata, heading hierarchy and other review signals. Inspect source-line evidence, suggested fixes and experimental model outputs.",
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

    # Interactive trend explorer: lets users compare key findings across audits.
    if st.session_state.ax_runs:
        section("Session insights", "INTERACTIVE SCAN TREND")
        with st.container(border=True):
            trend_col, info_col = st.columns([1.4, 1], gap="large")
            with trend_col:
                trend_df = pd.DataFrame(st.session_state.ax_runs).copy().head(12).iloc[::-1]
                trend_options = {
                    "Missing alt attributes": "Missing alt",
                    "Links without accessible names": "Empty links",
                    "Unlabeled controls": "Unlabeled buttons",
                    "Experimental risk estimate (%)": "Risk estimate",
                }
                chosen_trend = st.selectbox(
                    "Metric to compare",
                    list(trend_options.keys()),
                    key="ax_overview_trend_metric",
                )
                trend_field = trend_options[chosen_trend]
                trend_df[trend_field] = pd.to_numeric(trend_df[trend_field], errors="coerce").fillna(0)
                trend_df["Scan"] = [f"Scan {i + 1}" for i in range(len(trend_df))]
                trend_fig = go.Figure(go.Scatter(
                    x=trend_df["Scan"],
                    y=trend_df[trend_field],
                    mode="lines+markers",
                    line={"color": "#69F0E4", "width": 3, "shape": "spline"},
                    marker={"size": 8, "color": "#8492FF", "line": {"color": "#DDFBF8", "width": 1}},
                    customdata=trend_df[["Target", "Time"]].to_numpy(),
                    hovertemplate="%{x}<br>Value: %{y}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                ))
                trend_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin={"l": 8, "r": 12, "t": 12, "b": 8},
                    height=260,
                    showlegend=False,
                    xaxis={"title": None, "showgrid": False, "tickfont": {"color": "#8797B3"}},
                    yaxis={"title": chosen_trend, "gridcolor": "rgba(145,163,200,.13)", "rangemode": "tozero"},
                    hoverlabel={"bgcolor": "#101827", "bordercolor": "#34435e", "font": {"color": "#eef3ff"}},
                )
                st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False}, key="ax_overview_trend_chart")
            with info_col:
                st.markdown("### Compare your scans")
                st.caption("Choose a metric to see how it changes across your most recent audits.")
                latest_value = trend_df[trend_field].iloc[-1] if not trend_df.empty else 0
                previous_value = trend_df[trend_field].iloc[-2] if len(trend_df) > 1 else None
                st.metric("Latest scan", f"{latest_value:.1f}" if trend_field == "Risk estimate" else f"{int(latest_value)}")
                if previous_value is not None:
                    delta = latest_value - previous_value
                    st.metric("Change vs. previous scan", f"{delta:+.1f}" if trend_field == "Risk estimate" else f"{delta:+.0f}")
                st.markdown('<div class="ax-callout">This trend reflects only the current Streamlit session. Risk estimates are experimental model outputs, not WCAG scores or certification.</div>', unsafe_allow_html=True)

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
                ("02", "Inspect structure", "Check image text, links, controls, forms, headings, frames, IDs, language and title."),
                ("03", "Explain & report", "Show evidence, recommend fixes, and export findings for follow-up."),
            ]:
                st.markdown(
                    f'<div class="ax-list-row"><div style="display:flex;gap:12px;align-items:center"><span style="color:#70e9df;font:12px DM Mono,monospace">{n}</span><div><div class="ax-list-title">{title}</div><div class="ax-list-sub">{desc}</div></div></div><span class="ax-tag">PIPELINE</span></div>',
                    unsafe_allow_html=True,
                )
    if st.session_state.ax_last_audit:
        show_audit_results(st.session_state.ax_last_audit, rf_model)
    st.markdown('<div class="ax-footer"><span>ACCESSIFY MK2 / ACCESSIBILITY INTELLIGENCE</span><span>PROTOTYPE · SESSION-BASED HISTORY</span></div>', unsafe_allow_html=True)


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


# --- Models & data -------------------------------------------------------------
elif page == "Models & data":
    hero(
        "MODEL DIAGNOSTICS",
        'Know what powers <span class="ax-gradient">the audit.</span>',
        "Explore the real violation dataset, run a URL-grouped holdout evaluation, inspect feature signals, and see which issue categories dominate the corpus.",
    )
    df = load_dataset_dataframe(csv_bytes)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset records", f"{len(df):,}" if not df.empty else "Unavailable")
    m2.metric("Unique URLs", f"{df['webURL'].nunique():,}" if not df.empty and "webURL" in df else "—")
    m3.metric("Distinct descriptions", f"{df['Description'].nunique():,}" if not df.empty and "Description" in df else "—")
    m4.metric("RF trees", int(rf_model.n_estimators))

    section("Training source & schema", "LIVE DATASET INSPECTION")
    with st.container(border=True):
        st.write(model_source)
        if not df.empty:
            st.write("**Columns found:** " + " · ".join(str(c) for c in df.columns))
            st.caption("Impact labels are derived from initialImpactScore ≥ 5. They reflect this dataset's scoring scheme; they are not verified real-world accessibility outcomes.")
        else:
            st.warning("No compatible dataset found. Add Original_full_data (1).csv beside app.py or upload a CSV in the sidebar.")
        st.markdown('<div class="ax-callout"><b>Evaluation safeguard:</b> the held-out model below keeps records from each webURL in one split to reduce page-level leakage. Metrics still describe prediction of the dataset&#39;s impact-score label—not WCAG conformance.</div>', unsafe_allow_html=True)

    section("URL-grouped model evaluation", "TEXT-BASED EXPERIMENT")
    with st.container(border=True):
        with st.spinner("Evaluating text model on a held-out set of URL groups…"):
            evaluation = evaluate_dataset_model(csv_bytes)
        if not evaluation.get("available"):
            st.info(evaluation.get("reason", "Metrics unavailable for this data."))
        else:
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Accuracy", f"{evaluation['accuracy']:.1%}")
            e2.metric("Precision", f"{evaluation['precision']:.1%}")
            e3.metric("Recall", f"{evaluation['recall']:.1%}")
            e4.metric("F1 score", f"{evaluation['f1']:.1%}")
            st.caption(f"{evaluation['train_rows']:,} training records across {evaluation['train_urls']:,} URLs · {evaluation['test_rows']:,} test records across {evaluation['test_urls']:,} unseen URLs. Target: impact score ≥ 5. Split: GroupShuffleSplit by webURL, random_state=42. Repeated violation descriptions can still appear across different URLs, so these metrics may be optimistic and should be treated as a corpus-label experiment.")
            cm = evaluation["confusion_matrix"]
            cmfig = go.Figure(go.Heatmap(z=cm, x=["Predicted lower", "Predicted higher"], y=["Actual lower", "Actual higher"], colorscale=[[0,"#101827"],[0.5,"#4c6c95"],[1,"#69f0e4"]], text=cm, texttemplate="%{text}", hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>"))
            cmfig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8,r=8,t=15,b=8), height=280, xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"))
            c1,c2=st.columns([1,1.3], gap="large")
            with c1:
                st.markdown("**Confusion matrix**")
                st.plotly_chart(cmfig, use_container_width=True, config={"displayModeBar":False}, key="dataset_confusion_matrix")
            with c2:
                st.markdown("**Strongest text-model terms**")
                term_df = pd.DataFrame(evaluation["top_terms"])
                st.dataframe(term_df, hide_index=True, use_container_width=True, height=300)
            st.download_button("Download evaluation summary JSON", json.dumps(evaluation, indent=2, default=str).encode("utf-8"), "accessify_model_evaluation.json", "application/json")

    if not df.empty:
        section("Dataset explorer", "CATEGORY / IMPACT / DOMAIN")
        chart1, chart2 = st.columns([1, 1], gap="large")
        with chart1:
            if "initialImpactScore" in df:
                impact = pd.to_numeric(df["initialImpactScore"], errors="coerce").dropna().value_counts().sort_index().reset_index()
                impact.columns = ["Impact score", "Records"]
                fig = go.Figure(go.Bar(x=impact["Impact score"].astype(str), y=impact["Records"], marker_color=["#8492ff","#8492ff","#69f0e4","#ffbf69"][:len(impact)], hovertemplate="Score %{x}<br>%{y} records<extra></extra>"))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=15,b=5), height=300, xaxis_title="Initial impact score", yaxis_title="Records", yaxis=dict(gridcolor="rgba(145,163,200,.13)"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False}, key="impact_distribution")
        with chart2:
            if "category" in df and "domain" in df:
                pivot = pd.crosstab(df["domain"].fillna("Unknown"), df["category"].fillna("Unknown"))
                heat = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str), colorscale=[[0,"#101827"],[0.5,"#536d9b"],[1,"#69f0e4"]], hovertemplate="Domain: %{y}<br>Category: %{x}<br>Records: %{z}<extra></extra>"))
                heat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8,r=8,t=15,b=8), height=300, xaxis_title="Violation category", yaxis_title="Dataset domain")
                st.plotly_chart(heat, use_container_width=True, config={"displayModeBar":False}, key="dataset_domain_heatmap")
        dleft, dright = st.columns([1.2, 1], gap="large")
        with dleft:
            st.markdown("**Most common violation descriptions**")
            if "Description" in df:
                top_desc = df["Description"].fillna("Unknown").value_counts().head(15).rename_axis("Description").reset_index(name="Records")
                st.dataframe(top_desc, hide_index=True, use_container_width=True, height=360)
        with dright:
            st.markdown("**Dataset preview**")
            preview_cols = [c for c in ["webURL", "Description", "initialImpactScore", "category", "domain", "fixed_code_desc"] if c in df]
            preview_df = df[preview_cols].head(25).copy()
            for text_col in ["webURL", "Description", "fixed_code_desc"]:
                if text_col in preview_df:
                    preview_df[text_col] = preview_df[text_col].fillna("").astype(str).map(lambda value: value[:240] + ("…" if len(value) > 240 else ""))
            st.dataframe(preview_df, hide_index=True, use_container_width=True, height=360)
            st.download_button("Download filtered/source dataset", df.to_csv(index=False).encode("utf-8"), "accessify_dataset_export.csv", "text/csv")

    section("Live audit model signals", "GLOBAL RANDOM FOREST FEATURE IMPORTANCE")
    feature_importances = {
        "Missing alt attributes": float(rf_model.feature_importances_[0]),
        "Links without accessible names": float(rf_model.feature_importances_[1]),
        "Unlabeled buttons": float(rf_model.feature_importances_[2]),
    }
    render_feature_importance_chart(feature_importances, key="model_data_feature_importance")
    section("Model details", "LIMITATIONS")
    with st.container(border=True):
        st.markdown("**Live-page risk estimator:** Random Forest over three structural counts. The model is trained on URL-level aggregates derived from violation descriptions, and its prediction is experimental.")
        st.markdown("**Text model:** TF-IDF + Logistic Regression on dataset descriptions. It predicts a threshold derived from `initialImpactScore`; this is not an alt-text quality classifier.")
        st.markdown("**Alt-text NLP signal:** the starter classifier uses hand-authored examples and is experimental; its predictions are suggestions for review, not validated accessibility judgments.")


# --- Remediation assistant ------------------------------------------------------
elif page == "Remediation assistant":
    hero(
        "REMEDIATION WORKBENCH",
        'Turn findings into <span class="ax-gradient">next steps.</span>',
        "Inspect the latest audit's evidence, see the source line and get a copyable starter correction. Suggestions are contextual examples and should be reviewed before applying them to production code.",
    )
    result = st.session_state.ax_last_audit
    if not result:
        st.info("Run a website audit first, or open Reports & demo to load the built-in sample page without internet access.")
    else:
        findings_df = pd.DataFrame(result.get("audit_findings", []))
        if findings_df.empty:
            st.success("No deterministic findings are currently recorded for this page. Automated checks are not a guarantee of compliance.")
        else:
            filters = st.columns([1,1,1])
            severity = filters[0].selectbox("Severity", ["All"] + sorted(findings_df["Severity"].dropna().unique().tolist()), key="rem_severity")
            category = filters[1].selectbox("Category", ["All"] + sorted(findings_df["Category"].dropna().unique().tolist()), key="rem_category")
            search = filters[2].text_input("Find an issue", placeholder="e.g. label", key="rem_search")
            filtered = findings_df.copy()
            if severity != "All": filtered = filtered[filtered["Severity"] == severity]
            if category != "All": filtered = filtered[filtered["Category"] == category]
            if search.strip(): filtered = filtered[filtered["Issue"].str.contains(search.strip(), case=False, regex=False)]
            st.dataframe(filtered[["Severity", "Category", "Issue", "Source line", "Element"]], hide_index=True, use_container_width=True, height=260)
            if len(filtered):
                selected_i = st.selectbox("Open remediation detail", list(range(len(filtered))), format_func=lambda i: f"Line {filtered.iloc[i]['Source line']} · {filtered.iloc[i]['Issue']}", key="rem_selected")
                item = filtered.iloc[int(selected_i)].to_dict()
                left, right = st.columns([1,1], gap="large")
                with left:
                    with st.container(border=True):
                        st.markdown(f"### {item['Issue']}")
                        st.caption(f"{item['Severity']} · {item['Category']} · source line {item['Source line']}")
                        st.write(item["Details"])
                        st.markdown("**Captured HTML evidence**")
                        st.code(str(item.get("HTML evidence", "")), language="html")
                with right:
                    with st.container(border=True):
                        st.markdown("### Suggested correction")
                        st.caption("Template only — check the real element, content, and design system.")
                        suggestion = str(item.get("Suggested fix", ""))
                        st.code(suggestion, language="html")
                        st.download_button("Download fix snippet", suggestion.encode("utf-8"), "accessify_fix_snippet.html", "text/html", type="primary", key="rem_download")
                        dataset_for_fixes = load_dataset_dataframe(csv_bytes)
                        if not dataset_for_fixes.empty and {"Description", "fixed_code_desc"}.issubset(dataset_for_fixes.columns):
                            issue_lower = str(item.get("Issue", "")).lower()
                            detail_lower = str(item.get("Details", "")).lower()
                            if "image" in issue_lower or "alt" in issue_lower:
                                match_terms = r"alt|alternative text|image"
                            elif "link" in issue_lower:
                                match_terms = r"link|discernible text|accessible name"
                            elif "button" in issue_lower or "control" in issue_lower or "label" in issue_lower:
                                match_terms = r"button|label|form control|discernible text|accessible name"
                            elif "frame" in issue_lower:
                                match_terms = r"frame|iframe"
                            elif "heading" in issue_lower:
                                match_terms = r"heading|header"
                            elif "language" in issue_lower or "title" in issue_lower:
                                match_terms = r"language|document title|title"
                            elif "id" in issue_lower:
                                match_terms = r"id attribute|unique id|duplicate id"
                            else:
                                match_terms = re.escape(issue_lower.split()[0]) if issue_lower.split() else r"accessibility"
                            desc_series = dataset_for_fixes["Description"].fillna("").astype(str)
                            code_series = dataset_for_fixes["fixed_code_desc"].fillna("").astype(str)
                            related = dataset_for_fixes[desc_series.str.contains(match_terms, case=False, regex=True, na=False) & code_series.str.strip().ne("")].copy()
                            if not related.empty:
                                with st.expander("Related remediation examples from your dataset", expanded=False):
                                    related = related.head(25).reset_index(drop=True)
                                    selected_example = st.selectbox("Select a related dataset example", list(range(len(related))), format_func=lambda j: str(related.iloc[j].get("Description", "Dataset example")), key="rem_dataset_example")
                                    example = related.iloc[int(selected_example)]
                                    st.caption("Dataset-provided example; relevance is keyword-based and may not match the current page exactly.")
                                    if pd.notna(example.get("updated_Description")):
                                        st.markdown("**Dataset explanation**")
                                        st.write(str(example.get("updated_Description"))[:5000])
                                    if pd.notna(example.get("AffectedHTMLElement(s)")):
                                        st.markdown("**Affected HTML example**")
                                        st.code(str(example.get("AffectedHTMLElement(s)"))[:5000], language="html")
                                    st.markdown("**Dataset remediation guidance**")
                                    st.markdown(str(example.get("fixed_code_desc", ""))[:7000])
                        if st.button("Add to remediation checklist", use_container_width=True, key="rem_add"):
                            st.session_state.setdefault("ax_remediation_checklist", [])
                            row = f"Line {item['Source line']}: {item['Issue']}"
                            if row not in st.session_state.ax_remediation_checklist:
                                st.session_state.ax_remediation_checklist.append(row)
                            st.success("Added to your session checklist.")
            checklist = st.session_state.get("ax_remediation_checklist", [])
            if checklist:
                section("Remediation checklist", "SESSION WORK ITEMS")
                st.write("\n".join(f"- [ ] {entry}" for entry in checklist))
                st.download_button("Export checklist", "\n".join(f"- [ ] {entry}" for entry in checklist).encode("utf-8"), "accessify_remediation_checklist.md", "text/markdown")


# --- Reports and demo mode ------------------------------------------------------
elif page == "Reports & demo":
    hero(
        "REPORTS & RELIABLE DEMO",
        'Show the work. <span class="ax-gradient">Even offline.</span>',
        "Run the built-in sample HTML without making a network request, inspect a realistic result, and export findings as CSV, JSON, or a shareable HTML report.",
    )
    demo_left, demo_right = st.columns([1,1], gap="large")
    with demo_left:
        with st.container(border=True):
            st.markdown("### Offline demo audit")
            st.write("Uses a bundled sample page with deliberately varied markup to exercise missing alt, vague links, unnamed buttons, form labels, heading hierarchy, frame titles, and empty disclosure labels.")
            if st.button("▶ Run offline demo audit", type="primary", use_container_width=True, key="ax_run_demo"):
                demo_result = run_demo_audit(vectorizer, nlp_model, rf_model)
                add_history(demo_result)
                st.session_state.ax_demo_active = True
                st.success("Demo audit complete; no outbound request was made.")
    with demo_right:
        with st.container(border=True):
            st.markdown("### What this demo proves")
            st.markdown("- Source-line evidence and HTML snippets\n- Structured issue severity and remediation suggestions\n- Interactive finding distribution charts\n- Downloadable exports for judges")
            st.caption("Demo results use sample HTML and should be presented as a feature demonstration, not a scan of a public website.")
    result = st.session_state.ax_last_audit
    if result:
        section("Current report", "LAST AUDIT IN THIS SESSION")
        st.write(f"**{result.get('page_title','Untitled')}** · {result.get('final_url', result.get('target',''))}")
        with st.container(border=True):
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Findings", len(result.get("audit_findings", [])))
            m2.metric("Missing alt", result.get("missing_alt", 0))
            m3.metric("Unnamed links", result.get("empty_links", 0))
            m4.metric("Unnamed controls", result.get("unlabeled_buttons", 0))
            findings = pd.DataFrame(result.get("audit_findings", []))
            if not findings.empty:
                st.dataframe(findings[["Severity","Category","Issue","Source line","Details"]], hide_index=True, use_container_width=True, height=300)
            json_payload = json.dumps(result, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            st.download_button("Download full report JSON", json_payload, "accessify_audit_report.json", "application/json", type="primary")
            if not findings.empty:
                st.download_button("Download all findings CSV", findings.to_csv(index=False).encode("utf-8"), "accessify_findings.csv", "text/csv")
            st.download_button("Download HTML report", build_report_html(result).encode("utf-8"), "accessify_audit_report.html", "text/html")
            semantic_df = pd.DataFrame(result.get("semantic_rows", []))
            if not semantic_df.empty:
                st.download_button("Download image semantic findings CSV", semantic_df.to_csv(index=False).encode("utf-8"), "accessify_semantic_findings.csv", "text/csv")
    else:
        st.info("No result is loaded yet. Run the offline demo or perform a live audit first.")


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
        "Set scan defaults.",
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

