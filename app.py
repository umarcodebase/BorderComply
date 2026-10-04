import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Dict

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from crewai import Agent, Crew, Process, Task, LLM

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "logo.jpg"
RESOURCES_PATH = BASE_DIR / "resources.txt"
DEFAULT_MODEL = "gemini/gemini-3.8-flash"

st.set_page_config(
    page_title="BorderComply | Pakistan",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { background: linear-gradient(180deg, #f6fbfa 0%, #ffffff 35%, #f7fafc 100%); }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 4rem; }
    .hero {
        border-radius: 26px; padding: 28px 32px 30px 32px;
        background: linear-gradient(135deg, #0b3b5e 0%, #123f64 55%, #087f63 100%);
        color: white; box-shadow: 0 18px 45px rgba(13, 53, 77, .15);
        margin-bottom: 22px;
    }
    .hero-eyebrow { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; opacity: .78; font-weight: 700; }
    .hero-title { font-size: 3rem; font-weight: 800; line-height: 1.03; margin: 9px 0 10px 0; }
    .hero-sub { font-size: 1.04rem; max-width: 780px; line-height: 1.6; opacity: .91; }
    .pill { display:inline-block; padding: 6px 10px; border-radius:999px; background:rgba(255,255,255,.12); margin-right:7px; font-size:.78rem; }
    .section-title { font-size: 1.32rem; font-weight: 800; color:#12334d; margin: 24px 0 12px; }
    .agent-card { border:1px solid #dfe8e7; background:#fff; border-radius:16px; padding:15px 16px; min-height:105px; box-shadow:0 5px 16px rgba(20,50,60,.04); }
    .agent-num { font-size:.72rem; color:#0a7f65; font-weight:800; }
    .agent-title { font-weight:750; color:#173b50; margin-top:4px; }
    .agent-text { color:#6a7781; font-size:.82rem; line-height:1.45; margin-top:5px; }
    .kpi { border:1px solid #e2e9e7; border-radius:16px; padding:14px 16px; background:white; }
    .kpi-label { color:#78858e; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .kpi-value { color:#15364e; font-size:1.6rem; font-weight:800; margin-top:3px; }
    .finding { border:1px solid #e5ecec; border-left:5px solid #0a7f65; background:#fff; border-radius:13px; padding:14px 16px; margin:8px 0; }
    .finding.warn { border-left-color:#c58a00; }
    .finding.risk { border-left-color:#b34242; }
    .source-card { border:1px solid #e4eae9; border-radius:14px; padding:13px 15px; background:#fbfdfd; margin:7px 0; }
    .source-id { font-size:.72rem; font-weight:800; color:#087f63; }
    .source-title { font-weight:700; color:#203d50; }
    .source-url { color:#55727a; font-size:.77rem; word-break:break-all; }
    .disclaimer { border:1px solid #e9dfb8; background:#fffdf2; border-radius:14px; padding:14px 16px; color:#685c30; font-size:.8rem; line-height:1.55; }
    .small-note { color:#71808a; font-size:.78rem; line-height:1.5; }
    div[data-testid="stFileUploader"] section { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Source registry
# -----------------------------

def parse_resources(path: Path) -> Dict[str, Dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    registry: Dict[str, Dict[str, str]] = {}
    current = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            registry[current] = {}
        elif current and "=" in line:
            key, value = line.split("=", 1)
            registry[current][key.strip()] = value.strip()
    return registry

SOURCES = parse_resources(RESOURCES_PATH)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_source(source_id: str) -> Dict[str, str]:
    src = SOURCES[source_id]
    url = src["URL"]
    try:
        r = requests.get(
            url,
            timeout=18,
            headers={"User-Agent": "BorderComply-MVP/1.0 (+trade-compliance-demo)"},
        )
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type or "application/xhtml" in content_type:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            text = " ".join(soup.stripped_strings)
        else:
            text = r.text
        text = re.sub(r"\s+", " ", text).strip()
        return {"ok": "true", "text": text[:7000], "retrieved_at": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        return {"ok": "false", "text": "", "error": str(exc), "retrieved_at": datetime.now(timezone.utc).isoformat()}


def fetch_source_pack(source_ids: List[str]) -> str:
    chunks = []
    for sid in source_ids:
        src = SOURCES[sid]
        result = fetch_source(sid)
        if result.get("ok") == "true" and result.get("text"):
            chunks.append(
                f"\n===== {sid} | {src['TITLE']} | {src['URL']} =====\n"
                f"Authority: {src['AUTHORITY']}\n"
                f"Scope: {src['SCOPE']}\n"
                f"Retrieved: {result['retrieved_at']}\n"
                f"Page content excerpt:\n{result['text']}\n"
            )
        else:
            chunks.append(
                f"\n===== {sid} | {src['TITLE']} | {src['URL']} =====\n"
                f"Authority: {src['AUTHORITY']}\nScope: {src['SCOPE']}\n"
                f"FETCH FAILED. Do NOT use this source as evidence.\n"
            )
    return "\n".join(chunks)


# -----------------------------
# PDF extraction
# -----------------------------

def extract_pdf_text(uploaded_files) -> str:
    parts = []
    for f in uploaded_files or []:
        try:
            reader = PdfReader(io.BytesIO(f.getvalue()))
            pages = []
            for page in reader.pages[:12]:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages).strip()
            if text:
                parts.append(f"===== USER DOCUMENT: {f.name} =====\n{text[:18000]}")
        except Exception as exc:
            parts.append(f"===== USER DOCUMENT: {f.name} =====\nText extraction failed: {exc}")
    return "\n\n".join(parts)


# -----------------------------
# CrewAI structured output
# -----------------------------

class ActionItem(BaseModel):
    title: str = Field(min_length=2, max_length=160)
    status: Literal["REQUIRED", "CHECK", "NOT_IDENTIFIED"]
    rationale: str = Field(min_length=2, max_length=500)
    source_ids: List[str] = Field(default_factory=list, max_length=5)

class DocumentItem(BaseModel):
    document: str = Field(min_length=2, max_length=140)
    purpose: str = Field(min_length=2, max_length=350)
    source_ids: List[str] = Field(default_factory=list, max_length=5)

class RiskItem(BaseModel):
    risk: str = Field(min_length=2, max_length=160)
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str = Field(min_length=2, max_length=450)
    source_ids: List[str] = Field(default_factory=list, max_length=5)

class SourceUse(BaseModel):
    source_id: str = Field(min_length=2, max_length=8)
    supports: str = Field(min_length=2, max_length=250)

class ComplianceReport(BaseModel):
    executive_summary: str = Field(min_length=20, max_length=1100)
    likely_hs_code: str = Field(min_length=2, max_length=120)
    hs_confidence: Literal["HIGH", "MEDIUM", "LOW", "NOT_DETERMINED"]
    actions: List[ActionItem] = Field(default_factory=list, max_length=8)
    documents: List[DocumentItem] = Field(default_factory=list, max_length=8)
    risks: List[RiskItem] = Field(default_factory=list, max_length=6)
    source_uses: List[SourceUse] = Field(default_factory=list, max_length=10)
    next_steps: List[str] = Field(default_factory=list, max_length=5)
    disclaimer: str = Field(min_length=30, max_length=700)


def build_crew(api_key: str):
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    llm = LLM(model=model_name, api_key=api_key, max_output_tokens=5000, timeout=120, max_retries=2)

    product_agent = Agent(
        role="Trade Product Analyst",
        goal="Normalise the user's product and identify the minimum factual attributes needed for compliance analysis.",
        backstory=(
            "You are a meticulous trade-product analyst. You distinguish user-provided facts from inference. "
            "You never invent an HS/PCT code. When classification is uncertain, you explicitly say what extra product detail is needed."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    pakistan_agent = Agent(
        role="Pakistan Trade Regulation Researcher",
        goal="Map Pakistan-side trade obligations using only the official sources supplied in the source pack.",
        backstory=(
            "You are a Pakistan trade-compliance researcher. You work only from the supplied official government sources. "
            "Every regulatory statement must include source IDs such as [S1]. You never fill a knowledge gap by guessing."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    destination_agent = Agent(
        role="Destination Market Analyst",
        goal="Assess destination-market requirements only from the supplied official destination source(s).",
        backstory=(
            "You analyse market-entry rules conservatively. You clearly separate what is evidenced by the official destination source "
            "from what requires confirmation. You never invent tariffs, licences, bans or product standards."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    risk_agent = Agent(
        role="Compliance Risk Reviewer",
        goal="Challenge the previous findings, detect unsupported claims, missing evidence and high-risk assumptions.",
        backstory=(
            "You are a hostile-but-fair compliance reviewer. You try to disprove conclusions. "
            "If a claim has no source evidence, you downgrade it to CHECK/NOT_IDENTIFIED. You distinguish AI inference from an official determination."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    verifier_agent = Agent(
        role="Senior Verification & Report Agent",
        goal="Produce a concise, source-grounded pre-trade report that never presents an unsupported regulatory claim as fact.",
        backstory=(
            "You are the final gatekeeper. You use only evidence present in the source pack and the earlier agent work. "
            "Every REQUIRED/CHECK regulatory action and every identified risk must have at least one valid source ID when a source-backed claim is possible. "
            "You must include the disclaimer that this is decision support, not legal/customs advice."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return product_agent, pakistan_agent, destination_agent, risk_agent, verifier_agent


def get_google_api_key() -> str:
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GOOGLE_API_KEY", "")


def run_analysis(user_inputs: dict, source_pack: str, document_text: str) -> ComplianceReport:
    api_key = get_google_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured. Add it to Streamlit Cloud Secrets.")

    product_agent, pakistan_agent, destination_agent, risk_agent, verifier_agent = build_crew(api_key)

    base_context = f"""
USER TRADE CASE
----------------
Trade direction: {user_inputs['trade_direction']}
Product name: {user_inputs['product_name']}
Product category: {user_inputs['product_category']}
Product description: {user_inputs['product_description']}
Origin country: {user_inputs['origin_country']}
Destination country/market: {user_inputs['destination']}
HS/PCT code supplied by user: {user_inputs['hs_code'] or 'None'}
Approx. shipment value: {user_inputs['value'] or 'Not supplied'}

USER-PROVIDED DOCUMENT TEXT
----------------
{document_text or 'No document uploaded.'}

OFFICIAL SOURCE PACK
----------------
{source_pack}

STRICT EVIDENCE RULE
----------------
Allowed evidence source IDs are only the IDs appearing in the source pack. Do not invent URLs, regulations, SRO numbers, tariffs, licence names, standards, deadlines or fees.
When the source pack does not establish a point, say "Not established from supplied sources" or "CHECK with the relevant authority".
"""

    t1 = Task(
        description=base_context + """

Analyse the product and trade transaction. Extract known facts, ambiguities and classification-relevant attributes. Explain what can and cannot be determined from the user's information. Do not make regulatory claims.
""",
        expected_output="A concise product profile and list of classification/compliance information gaps.",
        agent=product_agent,
    )

    t2 = Task(
        description=base_context + """

Using the product analysis as context, research Pakistan-side requirements. Cover only requirements that are supported by the supplied official Pakistani sources. For each point, include the source ID in square brackets, e.g. [S1]. Distinguish Customs/PSW/process requirements from product standards or policy restrictions.
""",
        expected_output="Source-backed Pakistan trade compliance findings with source IDs and explicit uncertainty where evidence is incomplete.",
        agent=pakistan_agent,
        context=[t1],
    )

    t3 = Task(
        description=base_context + """

Using the product analysis and the selected destination source(s), assess destination-side requirements. If the destination lacks a dedicated official source in the pack, clearly say that the destination side is not fully assessed. Never substitute general world knowledge for missing evidence.
""",
        expected_output="Source-backed destination market findings with source IDs and uncertainty notes.",
        agent=destination_agent,
        context=[t1],
    )

    t4 = Task(
        description=base_context + """

Review the outputs from the product, Pakistan, and destination analyses as an adversarial verifier. Identify unsupported claims, ambiguous HS/PCT classification, missing documents, missing source support, and areas needing professional confirmation. Your job is to catch errors, not to add new facts from memory.
""",
        expected_output="A verification memo listing confirmed findings, unsupported claims to remove, evidence gaps, and high/medium/low risks.",
        agent=risk_agent,
        context=[t1, t2, t3],
    )

    final_task = Task(
        description=base_context + """

Create the final BorderComply report from all preceding work.

OUTPUT RULES:
1. This is a pre-trade intelligence report, not an official customs ruling or legal opinion.
2. Never state an exact tariff rate, licence, prohibition, certificate, standard, fee, or deadline unless supported by a source ID in the provided source pack.
3. Every regulatory action in actions and every risk that depends on regulation should include one or more source IDs.
4. If HS/PCT classification is uncertain, set hs_confidence to NOT_DETERMINED or LOW and explain the missing information in next_steps. Do not pretend the AI has made an official classification.
5. Keep user-provided facts separate from regulatory conclusions.
6. Use plain business language suitable for a Pakistani SME owner.
7. source_uses may include only source IDs that exist in the source pack.
8. The final disclaimer must explicitly state that BorderComply is decision-support and that official authorities/customs agents/legal professionals should be consulted for binding determinations.
""",
        expected_output="A validated structured BorderComply compliance report.",
        agent=verifier_agent,
        context=[t1, t2, t3, t4],
        output_pydantic=ComplianceReport,
    )

    crew = Crew(
        agents=[product_agent, pakistan_agent, destination_agent, risk_agent, verifier_agent],
        tasks=[t1, t2, t3, t4, final_task],
        process=Process.sequential,
        verbose=False,
        max_rpm=4,
        share_crew=False,
        cache=True,
    )

    result = crew.kickoff()
    if result.pydantic:
        return result.pydantic
    try:
        return ComplianceReport.model_validate(result.to_dict())
    except Exception as exc:
        raise RuntimeError(f"Structured report validation failed: {exc}")


# -----------------------------
# Report rendering
# -----------------------------

def source_info(sid: str):
    return SOURCES.get(sid, {"TITLE": "Unknown source", "AUTHORITY": "Unknown", "URL": ""})


def make_markdown(report: ComplianceReport, user_inputs: dict, generated_at: str) -> str:
    lines = [
        "# BorderComply — Pakistan Pre-Trade Compliance Report",
        f"Generated: {generated_at}",
        "",
        "## Trade case",
        f"- Direction: {user_inputs['trade_direction']}",
        f"- Product: {user_inputs['product_name']}",
        f"- Category: {user_inputs['product_category']}",
        f"- Origin: {user_inputs['origin_country']}",
        f"- Destination: {user_inputs['destination']}",
        f"- User-supplied HS/PCT: {user_inputs['hs_code'] or 'Not supplied'}",
        "",
        "## Executive assessment",
        report.executive_summary,
        "",
        "## HS/PCT classification (AI-assisted only)",
        f"{report.likely_hs_code} — confidence: {report.hs_confidence}",
        "",
        "## Required / recommended actions",
    ]
    for a in report.actions:
        ids = ", ".join(a.source_ids) if a.source_ids else "No source identified — verify"
        lines.append(f"- **{a.status} — {a.title}:** {a.rationale} ({ids})")
    lines += ["", "## Documents / evidence"]
    for d in report.documents:
        ids = ", ".join(d.source_ids) if d.source_ids else "No source identified — verify"
        lines.append(f"- **{d.document}:** {d.purpose} ({ids})")
    lines += ["", "## Risks"]
    for r in report.risks:
        ids = ", ".join(r.source_ids) if r.source_ids else "No source identified — verify"
        lines.append(f"- **{r.severity} — {r.risk}:** {r.reason} ({ids})")
    lines += ["", "## Next steps"]
    for n in report.next_steps:
        lines.append(f"- {n}")
    lines += ["", "## Sources consulted"]
    for s in report.source_uses:
        info = source_info(s.source_id)
        lines.append(f"- **{s.source_id} — {info['TITLE']}** — {s.supports} — {info['URL']}")
    lines += ["", "## Disclaimer", report.disclaimer]
    return "\n".join(lines)


def make_pdf(report: ComplianceReport, user_inputs: dict, generated_at: str) -> bytes:
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, rightMargin=34, leftMargin=34, topMargin=34, bottomMargin=34)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BCHeader", parent=styles["Title"], fontSize=20, leading=24, textColor=colors.HexColor("#0d3b5e"), spaceAfter=8))
    styles.add(ParagraphStyle(name="BCSection", parent=styles["Heading2"], fontSize=12.5, leading=15, textColor=colors.HexColor("#087f63"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BCBody", parent=styles["BodyText"], fontSize=8.6, leading=12, textColor=colors.HexColor("#33434d")))
    styles.add(ParagraphStyle(name="BCTiny", parent=styles["BodyText"], fontSize=7.2, leading=9, textColor=colors.HexColor("#58666f")))

    story = [
        Paragraph("BorderComply", styles["BCHeader"]),
        Paragraph("Pakistan pre-trade compliance intelligence report", styles["BCBody"]),
        Spacer(1, 8),
        Paragraph(f"Generated: {generated_at}", styles["BCTiny"]),
        Spacer(1, 10),
    ]

    trade_data = [
        ["Direction", user_inputs["trade_direction"], "Product", user_inputs["product_name"]],
        ["Origin", user_inputs["origin_country"], "Destination", user_inputs["destination"]],
        ["Category", user_inputs["product_category"], "User HS/PCT", user_inputs["hs_code"] or "Not supplied"],
    ]
    tbl = Table(trade_data, colWidths=[62, 170, 66, 190])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f4f8f7")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#dfe7e4")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#233b4b")),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [tbl, Paragraph("Executive assessment", styles["BCSection"]), Paragraph(report.executive_summary, styles["BCBody"])]
    story += [Paragraph("HS/PCT classification — AI-assisted only", styles["BCSection"]), Paragraph(f"{report.likely_hs_code} (confidence: {report.hs_confidence})", styles["BCBody"])]

    story.append(Paragraph("Actions", styles["BCSection"]))
    for a in report.actions:
        ids = ", ".join(a.source_ids) if a.source_ids else "VERIFY"
        story.append(Paragraph(f"<b>{a.status} — {a.title}</b><br/>{a.rationale}<br/><font size='7'>Sources: {ids}</font>", styles["BCBody"]))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Documents / evidence", styles["BCSection"]))
    for d in report.documents:
        ids = ", ".join(d.source_ids) if d.source_ids else "VERIFY"
        story.append(Paragraph(f"<b>{d.document}</b> — {d.purpose} <font size='7'>({ids})</font>", styles["BCBody"]))
        story.append(Spacer(1, 3))

    story.append(PageBreak())
    story.append(Paragraph("Risk review", styles["BCSection"]))
    for r in report.risks:
        ids = ", ".join(r.source_ids) if r.source_ids else "VERIFY"
        story.append(Paragraph(f"<b>{r.severity} — {r.risk}</b><br/>{r.reason}<br/><font size='7'>Sources: {ids}</font>", styles["BCBody"]))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Next steps", styles["BCSection"]))
    for n in report.next_steps:
        story.append(Paragraph("• " + n, styles["BCBody"]))
        story.append(Spacer(1, 2))

    story.append(Paragraph("Sources consulted", styles["BCSection"]))
    for s in report.source_uses:
        info = source_info(s.source_id)
        story.append(Paragraph(f"<b>{s.source_id} — {info['TITLE']}</b><br/>{info['AUTHORITY']}<br/>{s.supports}<br/><font size='7'>{info['URL']}</font>", styles["BCTiny"]))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Disclaimer", styles["BCSection"]))
    story.append(Paragraph(report.disclaimer, styles["BCBody"]))
    doc.build(story)
    return out.getvalue()


# -----------------------------
# UI
# -----------------------------

if LOGO_PATH.exists():
    c1, c2 = st.columns([0.14, 0.86])
    with c1:
        st.image(str(LOGO_PATH), width=90)
    with c2:
        st.markdown(
            '<div style="padding-top:7px"><div style="font-size:1.55rem;font-weight:800;color:#0d3b5e">BorderComply</div>'
            '<div style="color:#5f7078;font-size:.92rem">AI pre-trade compliance intelligence for Pakistani importers & exporters</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="hero"><div class="hero-eyebrow">Pakistan-first • Source-grounded • Human-reviewed</div>'
    '<div class="hero-title">Know what you need<br>before you trade.</div>'
    '<div class="hero-sub">BorderComply turns product and trade information into a source-backed compliance checklist. '
    'It sits before Pakistan Single Window: helping a business understand the likely requirements, evidence and verification steps before filing or shipping.</div>'
    '<div style="margin-top:15px"><span class="pill">5-agent workflow</span><span class="pill">Official-source registry</span><span class="pill">AI-assisted classification</span><span class="pill">Downloadable report</span></div></div>',
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
    st.markdown("**Model**")
    st.code(os.getenv("GEMINI_MODEL", DEFAULT_MODEL), language="text")
    st.markdown('<div class="small-note">Default: Gemini 3.8 Flash. Google documents a free tier for this model, but quotas/availability vary by account and can change. Keep your key in Streamlit Secrets, never in GitHub.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Evidence rule**")
    st.markdown('<div class="small-note">Regulatory claims are required to cite the official source IDs available in <code>resources.txt</code>. Unsupported points are labelled for verification rather than invented.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">How BorderComply thinks</div>', unsafe_allow_html=True)
agent_cols = st.columns(5)
agent_cards = [
    ("01", "Product Analyst", "Normalises the product and identifies missing attributes."),
    ("02", "Pakistan Researcher", "Maps Pakistan-side requirements from official sources."),
    ("03", "Market Analyst", "Checks the selected destination source pack."),
    ("04", "Risk Reviewer", "Challenges weak evidence and hidden assumptions."),
    ("05", "Verification Agent", "Builds the final source-grounded report."),
]
for col, (num, title, desc) in zip(agent_cols, agent_cards):
    with col:
        st.markdown(f'<div class="agent-card"><div class="agent-num">AGENT {num}</div><div class="agent-title">{title}</div><div class="agent-text">{desc}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Official source coverage</div>', unsafe_allow_html=True)
source_cols = st.columns(4)
for i, sid in enumerate(["S1", "S3", "S4", "S6", "S7", "S8", "S9", "S11"]):
    info = SOURCES[sid]
    with source_cols[i % 4]:
        st.markdown(f'<div class="source-card"><div class="source-id">{sid}</div><div class="source-title">{info["TITLE"]}</div><div class="source-url">{info["AUTHORITY"]}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Run a compliance check</div>', unsafe_allow_html=True)
valid = product_name.strip() and product_description.strip() and origin_country.strip() and destination != "Other / not yet selected"

col_a, col_b = st.columns([1, 4])
with col_a:
    analyze = st.button("🔎 Analyze trade case", type="primary", use_container_width=True, disabled=not valid)
with col_b:
    if not valid:
        st.markdown('<div class="small-note">Complete the product name, description, origin and a supported destination to enable the analysis.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small-note">The app fetches the official source pack, then runs five sequential CrewAI agents. The final agent validates the structured result before it reaches the screen.</div>', unsafe_allow_html=True)

if analyze:
    destination_sources = ["S11"] if destination == "European Union" else ["S12"] if destination == "United Kingdom" else []
    pakistan_sources = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
    selected_sources = list(dict.fromkeys(pakistan_sources + destination_sources))

    with st.status("Running BorderComply's multi-agent verification…", expanded=True) as status:
        st.write("Fetching official source pack…")
        source_pack = fetch_source_pack(selected_sources)
        st.write("Extracting optional product documents…")
        document_text = extract_pdf_text(uploads)
        st.write("Product analysis → Pakistan research → market analysis → risk review → final verification")
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
        try:
            report = run_analysis(user_inputs, source_pack, document_text)
            status.update(label="Analysis complete — final report verified", state="complete", expanded=False)
            st.session_state["report"] = report
            st.session_state["inputs"] = user_inputs
            st.session_state["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception as exc:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(str(exc))
            st.info("On Streamlit Cloud, add GOOGLE_API_KEY under App settings → Secrets, then reboot the app if needed.")

report = st.session_state.get("report")
if report:
    generated_at = st.session_state.get("generated_at", "")
    inputs = st.session_state["inputs"]
    st.markdown('<div class="section-title">Compliance snapshot</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">HS/PCT confidence</div><div class="kpi-value">{report.hs_confidence}</div></div>', unsafe_allow_html=True)
    with k2:
        required = sum(1 for a in report.actions if a.status == "REQUIRED")
        st.markdown(f'<div class="kpi"><div class="kpi-label">Required actions</div><div class="kpi-value">{required}</div></div>', unsafe_allow_html=True)
    with k3:
        high = sum(1 for r in report.risks if r.severity == "HIGH")
        st.markdown(f'<div class="kpi"><div class="kpi-label">High risks</div><div class="kpi-value">{high}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Sources cited</div><div class="kpi-value">{len(report.source_uses)}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Executive assessment</div>', unsafe_allow_html=True)
    st.info(report.executive_summary)

    t1, t2, t3, t4 = st.tabs(["Actions", "Documents", "Risks", "Sources"])
    with t1:
        for a in report.actions:
            cls = "finding" if a.status == "REQUIRED" else "finding warn"
            src = ", ".join(a.source_ids) if a.source_ids else "No source — verify"
            st.markdown(f'<div class="{cls}"><b>{a.status} · {a.title}</b><br><span class="small-note">{a.rationale}</span><br><span class="source-id">Sources: {src}</span></div>', unsafe_allow_html=True)
    with t2:
        for d in report.documents:
            src = ", ".join(d.source_ids) if d.source_ids else "No source — verify"
            st.markdown(f'<div class="finding"><b>{d.document}</b><br><span class="small-note">{d.purpose}</span><br><span class="source-id">Sources: {src}</span></div>', unsafe_allow_html=True)
    with t3:
        for r in report.risks:
            cls = "finding risk" if r.severity == "HIGH" else "finding warn" if r.severity == "MEDIUM" else "finding"
            src = ", ".join(r.source_ids) if r.source_ids else "No source — verify"
            st.markdown(f'<div class="{cls}"><b>{r.severity} · {r.risk}</b><br><span class="small-note">{r.reason}</span><br><span class="source-id">Sources: {src}</span></div>', unsafe_allow_html=True)
    with t4:
        for s in report.source_uses:
            info = source_info(s.source_id)
            st.markdown(f'<div class="source-card"><div class="source-id">{s.source_id}</div><div class="source-title">{info["TITLE"]}</div><div class="small-note">{info["AUTHORITY"]}<br>{s.supports}<br><a href="{info["URL"]}" target="_blank">{info["URL"]}</a></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Next steps</div>', unsafe_allow_html=True)
    for n in report.next_steps:
        st.markdown(f"☑️ {n}")

    st.markdown('<div class="section-title">HS/PCT classification note</div>', unsafe_allow_html=True)
    st.warning(f"AI-assisted result: {report.likely_hs_code} · confidence: {report.hs_confidence}. BorderComply does not issue an official classification ruling.")

    st.markdown('<div class="section-title">Trust & limitations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">{report.disclaimer}</div>', unsafe_allow_html=True)

    md = make_markdown(report, inputs, generated_at)
    pdf = make_pdf(report, inputs, generated_at)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Download Markdown report", md, file_name="bordercomply_report.md", mime="text/markdown", use_container_width=True)
    with d2:
        st.download_button("⬇️ Download PDF report", pdf, file_name="bordercomply_report.pdf", mime="application/pdf", use_container_width=True)

    st.caption("BorderComply is a pre-trade decision-support prototype. Always verify binding requirements with the relevant authority, customs agent, or qualified trade/legal professional before a shipment.")
else:
    st.markdown('<div class="section-title">What the first version gives the user</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="agent-card"><div class="agent-num">INPUT</div><div class="agent-title">Product + trade details</div><div class="agent-text">Product, description, origin, destination, optional HS/PCT code and optional supporting PDFs.</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="agent-card"><div class="agent-num">INTELLIGENCE</div><div class="agent-title">Source-grounded reasoning</div><div class="agent-text">Five specialised CrewAI agents analyse, challenge and verify the case against the official source registry.</div></div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="agent-card"><div class="agent-num">OUTPUT</div><div class="agent-title">Actionable checklist</div><div class="agent-text">Requirements, documents, risks, next steps, HS/PCT confidence and source links in a downloadable report.</div></div>', unsafe_allow_html=True)
