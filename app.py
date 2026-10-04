import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape as esc
from pathlib import Path
from typing import Dict, List, Literal

import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "logo.jpg"
RESOURCES_PATH = BASE_DIR / "resources.txt"
DEFAULT_MODEL = "openai/gpt-oss-20b"   # Groq free tier, fastest, strict JSON schema support
SOURCE_CHARS = 1200                     # per official source page (keeps tokens low)
DOC_CHARS = 6000                        # total from uploaded PDFs

st.set_page_config(page_title="BorderComply | Pakistan", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Styling + animations
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { background: linear-gradient(180deg, #f6fbfa 0%, #ffffff 35%, #f7fafc 100%); }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 4rem; }

    @keyframes fadeUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;} }
    @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(10,127,101,.45)} 70%{box-shadow:0 0 0 10px rgba(10,127,101,0)} 100%{box-shadow:0 0 0 0 rgba(10,127,101,0)} }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes grow { from { width:0; } }
    @keyframes popIn { 0%{transform:scale(.6);opacity:0} 80%{transform:scale(1.08)} 100%{transform:scale(1);opacity:1} }

    .hero {
        border-radius: 26px; padding: 28px 32px 30px 32px;
        background: linear-gradient(135deg, #0b3b5e, #123f64, #087f63, #0b3b5e);
        background-size: 300% 300%; animation: gradientShift 14s ease infinite, fadeUp .6s ease both;
        color: white; box-shadow: 0 18px 45px rgba(13, 53, 77, .15); margin-bottom: 22px;
    }
    .hero-eyebrow { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; opacity: .78; font-weight: 700; }
    .hero-title { font-size: 3rem; font-weight: 800; line-height: 1.03; margin: 9px 0 10px 0; }
    .hero-sub { font-size: 1.04rem; max-width: 780px; line-height: 1.6; opacity: .91; }
    .pill { display:inline-block; padding: 6px 10px; border-radius:999px; background:rgba(255,255,255,.12); margin:0 7px 6px 0; font-size:.78rem; transition: background .2s; }
    .pill:hover { background:rgba(255,255,255,.24); }
    .section-title { font-size: 1.32rem; font-weight: 800; color:#12334d; margin: 24px 0 12px; animation: fadeUp .5s ease both; }

    .agent-card { border:1px solid #dfe8e7; background:#fff; border-radius:16px; padding:15px 16px; min-height:105px;
        box-shadow:0 5px 16px rgba(20,50,60,.04); animation: fadeUp .55s ease both; transition: transform .2s, box-shadow .2s, border-color .2s; }
    .agent-card:hover { transform: translateY(-4px); box-shadow:0 12px 26px rgba(20,50,60,.10); border-color:#9fd3c4; }
    .d1{animation-delay:.05s}.d2{animation-delay:.12s}.d3{animation-delay:.19s}.d4{animation-delay:.26s}.d5{animation-delay:.33s}
    .agent-num { font-size:.72rem; color:#0a7f65; font-weight:800; }
    .agent-title { font-weight:750; color:#173b50; margin-top:4px; }
    .agent-text { color:#6a7781; font-size:.82rem; line-height:1.45; margin-top:5px; }

    .kpi { border:1px solid #e2e9e7; border-radius:16px; padding:14px 16px; background:white; animation: popIn .5s ease both; transition: transform .2s; }
    .kpi:hover { transform: translateY(-3px); }
    .kpi-label { color:#78858e; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .kpi-value { color:#15364e; font-size:1.6rem; font-weight:800; margin-top:3px; }

    .finding { border:1px solid #e5ecec; border-left:5px solid #0a7f65; background:#fff; border-radius:13px; padding:14px 16px; margin:8px 0;
        animation: fadeUp .45s ease both; transition: transform .18s, box-shadow .18s; }
    .finding:hover { transform: translateX(4px); box-shadow:0 6px 18px rgba(20,50,60,.07); }
    .finding.warn { border-left-color:#c58a00; }
    .finding.risk { border-left-color:#b34242; }
    .source-card { border:1px solid #e4eae9; border-radius:14px; padding:13px 15px; background:#fbfdfd; margin:7px 0; animation: fadeUp .5s ease both; transition: border-color .2s, transform .2s; }
    .source-card:hover { border-color:#9fd3c4; transform: translateY(-2px); }
    .source-id { font-size:.72rem; font-weight:800; color:#087f63; }
    .source-title { font-weight:700; color:#203d50; }
    .source-url { color:#55727a; font-size:.77rem; word-break:break-all; }
    .disclaimer { border:1px solid #e9dfb8; background:#fffdf2; border-radius:14px; padding:14px 16px; color:#685c30; font-size:.8rem; line-height:1.55; }
    .small-note { color:#71808a; font-size:.78rem; line-height:1.5; }
    div[data-testid="stFileUploader"] section { border-radius: 14px; }
    .stButton button[kind="primary"] { transition: transform .15s, box-shadow .15s; }
    .stButton button[kind="primary"]:hover { transform: translateY(-2px); box-shadow:0 8px 20px rgba(8,127,99,.25); }

    /* live stepper */
    .stepper { border:1px solid #dfe8e7; background:#fff; border-radius:18px; padding:18px 20px; animation: fadeUp .4s ease both; }
    .bar { height:6px; border-radius:6px; background:#e8efee; overflow:hidden; margin-bottom:14px; }
    .bar > div { height:100%; background:linear-gradient(90deg,#0b3b5e,#0a7f65); transition: width .5s ease; animation: grow .5s ease; }
    .step { display:flex; align-items:center; gap:12px; padding:7px 0; color:#8a979f; font-size:.9rem; }
    .dot { width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.72rem; font-weight:800;
        border:2px solid #d5dfdd; color:#8a979f; flex-shrink:0; }
    .step.done { color:#173b50; } .step.done .dot { background:#0a7f65; border-color:#0a7f65; color:#fff; animation: popIn .35s ease; }
    .step.active { color:#0a7f65; font-weight:700; } .step.active .dot { border-color:#0a7f65; color:#0a7f65; animation: pulse 1.4s infinite; }
    .spin { width:12px; height:12px; border:2px solid #0a7f65; border-top-color:transparent; border-radius:50%; display:inline-block; animation: spin .8s linear infinite; margin-left:6px; vertical-align:middle; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Source registry
# -----------------------------

def parse_resources(path: Path) -> Dict[str, Dict[str, str]]:
    registry: Dict[str, Dict[str, str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            registry[current] = {}
        elif current and "=" in line:
            k, v = line.split("=", 1)
            registry[current][k.strip()] = v.strip()
    return registry

SOURCES = parse_resources(RESOURCES_PATH)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_source(source_id: str) -> Dict[str, str]:
    url = SOURCES[source_id]["URL"]
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "BorderComply-MVP/1.0 (+trade-compliance-demo)"})
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" in ctype:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.stripped_strings)
        else:
            text = r.text
        text = re.sub(r"\s+", " ", text).strip()
        return {"ok": "true", "text": text[:SOURCE_CHARS]}
    except Exception as exc:
        return {"ok": "false", "text": "", "error": str(exc)}


def fetch_source_pack(source_ids: List[str]) -> str:
    with ThreadPoolExecutor(max_workers=len(source_ids) or 1) as pool:
        results = list(pool.map(fetch_source, source_ids))
    chunks = []
    for sid, res in zip(source_ids, results):
        src = SOURCES[sid]
        head = f"[{sid}] {src['TITLE']} ({src['AUTHORITY']}) — scope: {src['SCOPE']}"
        body = res["text"] if res.get("ok") == "true" and res.get("text") else "FETCH FAILED — scope line only; do not cite page content."
        chunks.append(f"{head}\n{body}")
    return "\n\n".join(chunks)


def extract_pdf_text(uploaded_files) -> str:
    parts, budget = [], DOC_CHARS
    for f in uploaded_files or []:
        if budget <= 0:
            break
        try:
            reader = PdfReader(io.BytesIO(f.getvalue()))
            text = "\n".join((p.extract_text() or "") for p in reader.pages[:8])
            text = re.sub(r"\s+", " ", text).strip()[:budget]
            budget -= len(text)
            if text:
                parts.append(f"--- {f.name} ---\n{text}")
        except Exception as exc:
            parts.append(f"--- {f.name} --- (text extraction failed: {exc})")
    return "\n".join(parts)

# -----------------------------
# Structured output
# -----------------------------

class ActionItem(BaseModel):
    title: str
    status: Literal["REQUIRED", "CHECK", "NOT_IDENTIFIED"]
    rationale: str
    source_ids: List[str] = Field(default_factory=list)

class DocumentItem(BaseModel):
    document: str
    purpose: str
    source_ids: List[str] = Field(default_factory=list)

class RiskItem(BaseModel):
    risk: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str
    source_ids: List[str] = Field(default_factory=list)

class SourceUse(BaseModel):
    source_id: str
    supports: str

class ComplianceReport(BaseModel):
    executive_summary: str
    likely_hs_code: str
    hs_confidence: Literal["HIGH", "MEDIUM", "LOW", "NOT_DETERMINED"]
    actions: List[ActionItem] = Field(default_factory=list)
    documents: List[DocumentItem] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    source_uses: List[SourceUse] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    disclaimer: str

    @field_validator("source_uses")
    @classmethod
    def only_known_sources(cls, v):
        return [s for s in v if s.source_id in SOURCES]


def _arr(item_props: dict) -> dict:
    return {"type": "array", "items": {"type": "object", "properties": item_props,
            "required": list(item_props), "additionalProperties": False}}

_S = {"type": "string"}
_IDS = {"type": "array", "items": {"type": "string"}}
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": _S,
        "likely_hs_code": _S,
        "hs_confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW", "NOT_DETERMINED"]},
        "actions": _arr({"title": _S, "status": {"type": "string", "enum": ["REQUIRED", "CHECK", "NOT_IDENTIFIED"]}, "rationale": _S, "source_ids": _IDS}),
        "documents": _arr({"document": _S, "purpose": _S, "source_ids": _IDS}),
        "risks": _arr({"risk": _S, "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}, "reason": _S, "source_ids": _IDS}),
        "source_uses": _arr({"source_id": _S, "supports": _S}),
        "next_steps": {"type": "array", "items": _S},
        "disclaimer": _S,
    },
    "required": ["executive_summary", "likely_hs_code", "hs_confidence", "actions", "documents", "risks", "source_uses", "next_steps", "disclaimer"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are BorderComply, a Pakistan pre-trade compliance checker. Work through 5 steps internally, then output ONLY the final JSON report:
1 Product profiling: extract facts and classification-relevant gaps from the user's case; never invent an HS/PCT code.
2 Pakistan requirements: map Customs/PSW/OGA/policy obligations supported by the Pakistani sources.
3 Destination market: use only the destination source; if none is supplied, say the destination side is not assessed.
4 Risk review: challenge weak evidence; downgrade unsupported points to CHECK or NOT_IDENTIFIED.
5 Verification: produce a concise, source-grounded report.
RULES: Cite source IDs like S1 in source_ids. Never state tariffs, SRO numbers, licences, bans, standards, fees or deadlines unless the source text supports them; otherwise write "Not established from supplied sources — CHECK with authority". If HS is uncertain use LOW or NOT_DETERMINED. Plain business English for a Pakistani SME. Limits: actions<=6, documents<=6, risks<=5, source_uses<=8, next_steps<=4, each text field <=2 sentences, executive_summary <=4 sentences. The disclaimer must state this is decision-support, not a binding customs/legal determination, and that authorities, customs agents or legal professionals should be consulted."""


def get_secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, "")
    except Exception:
        val = ""
    return val or os.getenv(name, default)


def get_model() -> str:
    return get_secret("GROQ_MODEL", DEFAULT_MODEL)


def run_analysis(user_inputs: dict, source_pack: str, document_text: str) -> ComplianceReport:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to Streamlit Cloud → App settings → Secrets.")
    model = get_model()
    client = Groq(api_key=api_key, timeout=60, max_retries=2)

    user_msg = (
        f"CASE: direction={user_inputs['trade_direction']}; product={user_inputs['product_name']}; "
        f"category={user_inputs['product_category']}; origin={user_inputs['origin_country']}; "
        f"destination={user_inputs['destination']}; user_hs={user_inputs['hs_code'] or 'none'}; value={user_inputs['value'] or 'n/a'}\n"
        f"DESCRIPTION: {user_inputs['product_description']}\n\n"
        f"USER DOCUMENTS:\n{document_text or 'none'}\n\n"
        f"OFFICIAL SOURCES (only these IDs are valid):\n{source_pack}"
    )

    strict = model.startswith(("openai/gpt-oss", "qwen/"))
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_completion_tokens=2500,
    )
    if strict:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "compliance_report", "strict": True, "schema": REPORT_SCHEMA}}
    else:
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["messages"][0]["content"] += "\nReturn a JSON object with exactly these keys: " + json.dumps(REPORT_SCHEMA["properties"])
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    st.session_state["tokens"] = getattr(usage, "total_tokens", None)
    try:
        return ComplianceReport.model_validate_json(raw)
    except Exception as exc:
        raise RuntimeError(f"The model returned an invalid report. Please try again. ({exc.__class__.__name__})")

# -----------------------------
# Report rendering
# -----------------------------

def source_info(sid: str):
    return SOURCES.get(sid, {"TITLE": "Unknown source", "AUTHORITY": "Unknown", "URL": ""})


def ids_text(ids, empty="No source identified — verify"):
    return ", ".join(ids) if ids else empty


def make_markdown(report: ComplianceReport, u: dict, generated_at: str) -> str:
    lines = [
        "# BorderComply — Pakistan Pre-Trade Compliance Report", f"Generated: {generated_at}", "",
        "## Trade case",
        f"- Direction: {u['trade_direction']}", f"- Product: {u['product_name']}", f"- Category: {u['product_category']}",
        f"- Origin: {u['origin_country']}", f"- Destination: {u['destination']}", f"- User-supplied HS/PCT: {u['hs_code'] or 'Not supplied'}", "",
        "## Executive assessment", report.executive_summary, "",
        "## HS/PCT classification (AI-assisted only)", f"{report.likely_hs_code} — confidence: {report.hs_confidence}", "",
        "## Required / recommended actions",
    ]
    lines += [f"- **{a.status} — {a.title}:** {a.rationale} ({ids_text(a.source_ids)})" for a in report.actions]
    lines += ["", "## Documents / evidence"] + [f"- **{d.document}:** {d.purpose} ({ids_text(d.source_ids)})" for d in report.documents]
    lines += ["", "## Risks"] + [f"- **{r.severity} — {r.risk}:** {r.reason} ({ids_text(r.source_ids)})" for r in report.risks]
    lines += ["", "## Next steps"] + [f"- {n}" for n in report.next_steps]
    lines += ["", "## Sources consulted"]
    for s in report.source_uses:
        info = source_info(s.source_id)
        lines.append(f"- **{s.source_id} — {info['TITLE']}** — {s.supports} — {info['URL']}")
    lines += ["", "## Disclaimer", report.disclaimer]
    return "\n".join(lines)


def make_pdf(report: ComplianceReport, u: dict, generated_at: str) -> bytes:
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, rightMargin=34, leftMargin=34, topMargin=34, bottomMargin=34)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BCHeader", parent=styles["Title"], fontSize=20, leading=24, textColor=colors.HexColor("#0d3b5e"), spaceAfter=8))
    styles.add(ParagraphStyle(name="BCSection", parent=styles["Heading2"], fontSize=12.5, leading=15, textColor=colors.HexColor("#087f63"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BCBody", parent=styles["BodyText"], fontSize=8.6, leading=12, textColor=colors.HexColor("#33434d")))
    styles.add(ParagraphStyle(name="BCTiny", parent=styles["BodyText"], fontSize=7.2, leading=9, textColor=colors.HexColor("#58666f")))
    P = lambda txt, st_: Paragraph(txt, styles[st_])

    story = [P("BorderComply", "BCHeader"), P("Pakistan pre-trade compliance intelligence report", "BCBody"), Spacer(1, 8),
             P(f"Generated: {generated_at}", "BCTiny"), Spacer(1, 10)]
    trade_data = [
        ["Direction", u["trade_direction"], "Product", u["product_name"]],
        ["Origin", u["origin_country"], "Destination", u["destination"]],
        ["Category", u["product_category"], "User HS/PCT", u["hs_code"] or "Not supplied"],
    ]
    tbl = Table(trade_data, colWidths=[62, 170, 66, 190])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8f7")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dfe7e4")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#233b4b")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [tbl, P("Executive assessment", "BCSection"), P(esc(report.executive_summary), "BCBody")]
    story += [P("HS/PCT classification — AI-assisted only", "BCSection"), P(esc(f"{report.likely_hs_code} (confidence: {report.hs_confidence})"), "BCBody")]
    story.append(P("Actions", "BCSection"))
    for a in report.actions:
        story += [P(f"<b>{esc(a.status)} — {esc(a.title)}</b><br/>{esc(a.rationale)}<br/><font size='7'>Sources: {esc(ids_text(a.source_ids, 'VERIFY'))}</font>", "BCBody"), Spacer(1, 4)]
    story.append(P("Documents / evidence", "BCSection"))
    for d in report.documents:
        story += [P(f"<b>{esc(d.document)}</b> — {esc(d.purpose)} <font size='7'>({esc(ids_text(d.source_ids, 'VERIFY'))})</font>", "BCBody"), Spacer(1, 3)]
    story += [PageBreak(), P("Risk review", "BCSection")]
    for r in report.risks:
        story += [P(f"<b>{esc(r.severity)} — {esc(r.risk)}</b><br/>{esc(r.reason)}<br/><font size='7'>Sources: {esc(ids_text(r.source_ids, 'VERIFY'))}</font>", "BCBody"), Spacer(1, 4)]
    story.append(P("Next steps", "BCSection"))
    for n in report.next_steps:
        story += [P("• " + esc(n), "BCBody"), Spacer(1, 2)]
    story.append(P("Sources consulted", "BCSection"))
    for s in report.source_uses:
        info = source_info(s.source_id)
        story += [P(f"<b>{esc(s.source_id)} — {esc(info['TITLE'])}</b><br/>{esc(info['AUTHORITY'])}<br/>{esc(s.supports)}<br/><font size='7'>{esc(info['URL'])}</font>", "BCTiny"), Spacer(1, 4)]
    story += [P("Disclaimer", "BCSection"), P(esc(report.disclaimer), "BCBody")]
    doc.build(story)
    return out.getvalue()

# -----------------------------
# UI
# -----------------------------

STEPS = [
    ("01", "Product profiling", "Normalises the product and identifies missing attributes."),
    ("02", "Pakistan requirements", "Maps Pakistan-side obligations from official sources."),
    ("03", "Destination market", "Checks the selected destination source pack."),
    ("04", "Risk review", "Challenges weak evidence and hidden assumptions."),
    ("05", "Verification & report", "Builds the final source-grounded report."),
]


def render_stepper(placeholder, active: int, note: str = ""):
    """active = index of the running step (0-4); 5 = all done."""
    pct = int(min(active, 5) / 5 * 100)
    rows = []
    for i, (num, title, _) in enumerate(STEPS):
        cls = "done" if i < active else "active" if i == active else ""
        mark = "✓" if i < active else num
        spin = '<span class="spin"></span>' if i == active else ""
        rows.append(f'<div class="step {cls}"><div class="dot">{mark}</div><div>{title}{spin}</div></div>')
    note_html = f'<div class="small-note" style="margin-top:8px">{esc(note)}</div>' if note else ""
    placeholder.markdown(f'<div class="stepper"><div class="bar"><div style="width:{pct}%"></div></div>{"".join(rows)}{note_html}</div>', unsafe_allow_html=True)


if LOGO_PATH.exists():
    c1, c2 = st.columns([0.14, 0.86])
    with c1:
        st.image(str(LOGO_PATH), width=90)
    with c2:
        st.markdown(
            '<div style="padding-top:7px;animation:fadeUp .5s ease both"><div style="font-size:1.55rem;font-weight:800;color:#0d3b5e">BorderComply</div>'
            '<div style="color:#5f7078;font-size:.92rem">AI pre-trade compliance intelligence for Pakistani importers & exporters</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="hero"><div class="hero-eyebrow">Pakistan-first • Source-grounded • Human-reviewed</div>'
    '<div class="hero-title">Know what you need<br>before you trade.</div>'
    '<div class="hero-sub">BorderComply turns product and trade information into a source-backed compliance checklist. '
    'It sits before Pakistan Single Window: helping a business understand the likely requirements, evidence and verification steps before filing or shipping.</div>'
    '<div style="margin-top:15px"><span class="pill">5-step compliance check</span><span class="pill">Official-source registry</span>'
    '<span class="pill">AI-assisted classification</span><span class="pill">Results in seconds</span><span class="pill">Downloadable report</span></div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Trade case")
    trade_direction = st.selectbox("Trade direction", ["Export from Pakistan", "Import into Pakistan"])
    product_name = st.text_input("Product name", placeholder="e.g. Leather safety gloves")
    product_category = st.text_input("Product category", placeholder="e.g. PPE / industrial goods")
    product_description = st.text_area("Product description", height=120, placeholder="Material, intended use, technical features, packaging, size/model, etc.")
    origin_country = st.text_input("Origin country", value="Pakistan" if "Export" in trade_direction else "China")
    destination = st.selectbox("Destination market", ["European Union", "United Kingdom", "Pakistan", "Other / not yet selected"])
    hs_code = st.text_input("HS / PCT code (optional)", placeholder="Leave blank if unknown")
    value = st.text_input("Approx. shipment value (optional)", placeholder="e.g. USD 20,000")
    uploads = st.file_uploader("Product / commercial documents (optional)", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.markdown("**Evidence rule**")
    st.markdown('<div class="small-note">Regulatory claims must cite the official source IDs in <code>resources.txt</code>. Unsupported points are labelled for verification rather than invented.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">How the compliance check works</div>', unsafe_allow_html=True)
for i, (col, (num, title, desc)) in enumerate(zip(st.columns(5), STEPS)):
    with col:
        st.markdown(f'<div class="agent-card d{i+1}"><div class="agent-num">STEP {num}</div><div class="agent-title">{title}</div><div class="agent-text">{desc}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Official source coverage</div>', unsafe_allow_html=True)
source_cols = st.columns(4)
for i, sid in enumerate(["S1", "S3", "S4", "S6", "S7", "S8", "S9", "S11"]):
    info = SOURCES[sid]
    with source_cols[i % 4]:
        st.markdown(f'<div class="source-card" style="animation-delay:{i*0.05:.2f}s"><div class="source-id">{sid}</div><div class="source-title">{info["TITLE"]}</div><div class="source-url">{info["AUTHORITY"]}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Run a compliance check</div>', unsafe_allow_html=True)
valid = bool(product_name.strip() and product_description.strip() and origin_country.strip() and destination != "Other / not yet selected")

col_a, col_b = st.columns([1, 4])
with col_a:
    analyze = st.button("🔎 Analyze trade case", type="primary", use_container_width=True, disabled=not valid)
with col_b:
    if not valid:
        st.markdown('<div class="small-note">Complete the product name, description, origin and a supported destination to enable the analysis.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small-note">Official sources are gathered in parallel, then the 5-step check runs and the result is validated before it reaches the screen. Usually ready in a few seconds.</div>', unsafe_allow_html=True)

if analyze:
    destination_sources = ["S11"] if destination == "European Union" else ["S12"] if destination == "United Kingdom" else []
    selected_sources = list(dict.fromkeys(["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"] + destination_sources))
    user_inputs = {
        "trade_direction": trade_direction,
        "product_name": product_name.strip(),
        "product_category": product_category.strip() or "Not supplied",
        "product_description": product_description.strip(),
        "origin_country": origin_country.strip(),
        "destination": destination,
        "hs_code": hs_code.strip(),
        "value": value.strip(),
    }
    t0 = time.perf_counter()
    stepper = st.empty()
    render_stepper(stepper, 0, "Reading your product details and documents…")
    document_text = extract_pdf_text(uploads)
    render_stepper(stepper, 1, "Gathering official Pakistani and destination sources…")
    source_pack = fetch_source_pack(selected_sources)
    render_stepper(stepper, 2, "Checking requirements, reviewing risks and verifying the report…")
    try:
        report = run_analysis(user_inputs, source_pack, document_text)
        render_stepper(stepper, 5, f"Done in {time.perf_counter() - t0:.1f}s — report verified.")
        st.session_state.update(report=report, inputs=user_inputs,
                                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        st.toast("Compliance report ready", icon="✅")
    except Exception as exc:
        stepper.empty()
        msg = str(exc)
        if "429" in msg or "rate" in msg.lower():
            st.error("The free AI quota is busy right now. Please wait a minute and try again.")
        else:
            st.error(msg)
        st.info("On Streamlit Cloud, make sure GROQ_API_KEY is set under App settings → Secrets, then reboot the app if needed.")

report = st.session_state.get("report")
if report:
    generated_at = st.session_state.get("generated_at", "")
    inputs = st.session_state["inputs"]
    st.markdown('<div class="section-title">Compliance snapshot</div>', unsafe_allow_html=True)
    kpis = [
        ("HS/PCT confidence", report.hs_confidence),
        ("Required actions", sum(1 for a in report.actions if a.status == "REQUIRED")),
        ("High risks", sum(1 for r in report.risks if r.severity == "HIGH")),
        ("Sources cited", len(report.source_uses)),
    ]
    for i, (col, (label, val)) in enumerate(zip(st.columns(4), kpis)):
        with col:
            st.markdown(f'<div class="kpi" style="animation-delay:{i*0.08:.2f}s"><div class="kpi-label">{label}</div><div class="kpi-value">{esc(str(val))}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Executive assessment</div>', unsafe_allow_html=True)
    st.info(report.executive_summary)

    t1, t2, t3, t4 = st.tabs(["Actions", "Documents", "Risks", "Sources"])
    with t1:
        for i, a in enumerate(report.actions):
            cls = "finding" if a.status == "REQUIRED" else "finding warn"
            st.markdown(f'<div class="{cls}" style="animation-delay:{i*0.06:.2f}s"><b>{esc(a.status)} · {esc(a.title)}</b><br><span class="small-note">{esc(a.rationale)}</span><br><span class="source-id">Sources: {esc(ids_text(a.source_ids, "No source — verify"))}</span></div>', unsafe_allow_html=True)
    with t2:
        for i, d in enumerate(report.documents):
            st.markdown(f'<div class="finding" style="animation-delay:{i*0.06:.2f}s"><b>{esc(d.document)}</b><br><span class="small-note">{esc(d.purpose)}</span><br><span class="source-id">Sources: {esc(ids_text(d.source_ids, "No source — verify"))}</span></div>', unsafe_allow_html=True)
    with t3:
        for i, r in enumerate(report.risks):
            cls = "finding risk" if r.severity == "HIGH" else "finding warn" if r.severity == "MEDIUM" else "finding"
            st.markdown(f'<div class="{cls}" style="animation-delay:{i*0.06:.2f}s"><b>{esc(r.severity)} · {esc(r.risk)}</b><br><span class="small-note">{esc(r.reason)}</span><br><span class="source-id">Sources: {esc(ids_text(r.source_ids, "No source — verify"))}</span></div>', unsafe_allow_html=True)
    with t4:
        for s in report.source_uses:
            info = source_info(s.source_id)
            st.markdown(f'<div class="source-card"><div class="source-id">{esc(s.source_id)}</div><div class="source-title">{info["TITLE"]}</div><div class="small-note">{info["AUTHORITY"]}<br>{esc(s.supports)}<br><a href="{info["URL"]}" target="_blank">{info["URL"]}</a></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Next steps</div>', unsafe_allow_html=True)
    for n in report.next_steps:
        st.markdown(f"☑️ {n}")

    st.markdown('<div class="section-title">HS/PCT classification note</div>', unsafe_allow_html=True)
    st.warning(f"AI-assisted result: {report.likely_hs_code} · confidence: {report.hs_confidence}. BorderComply does not issue an official classification ruling.")

    st.markdown('<div class="section-title">Trust & limitations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">{esc(report.disclaimer)}</div>', unsafe_allow_html=True)

    md = make_markdown(report, inputs, generated_at)
    pdf = make_pdf(report, inputs, generated_at)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Download Markdown report", md, file_name="bordercomply_report.md", mime="text/markdown", use_container_width=True)
    with d2:
        st.download_button("⬇️ Download PDF report", pdf, file_name="bordercomply_report.pdf", mime="application/pdf", use_container_width=True)

    st.caption("BorderComply is a pre-trade decision-support prototype. Always verify binding requirements with the relevant authority, customs agent, or qualified trade/legal professional before a shipment.")
else:
    st.markdown('<div class="section-title">What you get</div>', unsafe_allow_html=True)
    cards = [
        ("INPUT", "Product + trade details", "Product, description, origin, destination, optional HS/PCT code and optional supporting PDFs."),
        ("CHECK", "5-step compliance check", "Your case is profiled, mapped to Pakistani and destination rules, risk-reviewed and verified against the official source registry."),
        ("OUTPUT", "Actionable checklist", "Requirements, documents, risks, next steps, HS/PCT confidence and source links in a downloadable report."),
    ]
    for i, (col, (tag, title, text)) in enumerate(zip(st.columns(3), cards)):
        with col:
            st.markdown(f'<div class="agent-card d{i+1}"><div class="agent-num">{tag}</div><div class="agent-title">{title}</div><div class="agent-text">{text}</div></div>', unsafe_allow_html=True)
