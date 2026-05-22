"""
Link3 Sales Funnel AI Closer v3
Powered by LM Studio (Local LLM)

This Streamlit demo shows how a small local LLM can do more than classify a
lead. It asks the model to produce a structured sales decision package:
score, buying signals, package fit, objection handling, follow-up plan, and
CRM notes. Version 3 tunes the sales logic for Link3 ISP operating realities.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODELS_URL = "http://localhost:1234/v1/models"
DEFAULT_MODEL = "qwen2.5-coder-1.5b-instruct"

FUNNEL_STAGES = ["Awareness", "Interest", "Consideration", "Intent", "Purchase", "Retention"]

NEXT_ACTIONS = {
    "Awareness": "Send area coverage, package range, and a simple fiber vs broadband explanation",
    "Interest": "Share BDT package options, installation charge note, and local success story",
    "Consideration": "Schedule a site survey or technical discussion with area feasibility check",
    "Intent": "Call today, confirm address, and send a BDT quotation",
    "Purchase": "Prepare onboarding, installation slot, billing, and required documents",
    "Retention": "Offer upgrade, backup link, static IP, or SLA support",
}

PACKAGES = {
    "B2B": [
        "Corporate Dedicated Internet",
        "SME Shared Fiber",
        "Backup Internet Link",
        "Static Public IP",
        "SLA Priority Support",
        "Multi-Branch Connectivity",
        "MikroTik / Router Support",
        "CCTV / IP Camera Remote Access",
        "IP Transit / Data Connectivity",
    ],
    "B2C": [
        "Home Fiber 30 Mbps",
        "Home Fiber 50 Mbps",
        "Home Fiber 100 Mbps",
        "Home Fiber 200 Mbps",
        "Gaming Low-Latency Package",
        "Streaming Friendly Package",
        "Work-from-Home Package",
        "Student Budget Package",
    ],
}

LINK3_MARKET_CONTEXT = {
    "areas": [
        "Banani",
        "Gulshan",
        "Baridhara",
        "Uttara",
        "Mirpur",
        "Dhanmondi",
        "Mohammadpur",
        "Bashundhara R/A",
        "Badda",
        "Motijheel",
        "Kawran Bazar",
        "Tejgaon",
        "Farmgate",
        "Wari",
        "Chattogram",
        "Sylhet",
        "Khulna",
        "Rajshahi",
    ],
    "customer_types": [
        "home user",
        "student",
        "freelancer",
        "small shop",
        "restaurant",
        "school",
        "clinic",
        "software company",
        "garments office",
        "call center",
        "corporate head office",
        "multi-branch business",
    ],
    "payment_channels": ["bKash", "Nagad", "Rocket", "bank transfer", "card", "cash collection"],
    "documents": ["NID", "trade license", "TIN/BIN", "office address", "authorized contact person"],
    "local_objections": [
        "installation charge",
        "monthly bill in BDT",
        "coverage in exact building",
        "real speed at night",
        "Facebook/YouTube/Netflix performance",
        "gaming ping",
        "support response time",
        "public IP price",
        "contract duration",
    ],
}

DEMO_LEADS = {
    "Hot B2B Banani lead": (
        "Hot B2B lead. 45 employees. Needs dedicated fiber in Banani. Timeline urgent. "
        "They need static public IP, Zoom stability, and a site survey before Sunday."
    ),
    "Multi-branch competitor risk": (
        "We are comparing Link3 with another ISP for our 3 branches in Gulshan, Motijheel, and Uttara. "
        "Need 500 Mbps dedicated fiber, static IP, SLA, and one bill. Please send BDT pricing by tomorrow."
    ),
    "Mirpur budget home user": (
        "ভাই, Mirpur DOHS এর বাসায় রাতের বেলা YouTube আর Facebook slow হয়. "
        "100 Mbps চাই, কিন্তু monthly budget 1200-1500 BDT. Installation charge কত?"
    ),
    "Existing SME upgrade": (
        "We already use Link3 at our Tejgaon office. Team size grew to 80 people and video calls lag. "
        "Need upgrade options, backup internet, MikroTik support, and better SLA."
    ),
    "Small shop awareness": (
        "Hello, checking internet packages for a small pharmacy near Dhanmondi 27. "
        "No urgent need, just send details and bKash payment options."
    ),
    "Freelancer gaming lead": (
        "আমি একজন freelancer, Bashundhara R/A তে থাকি. Upwork calls, file upload, and Valorant ping important. "
        "Need stable 50 or 100 Mbps line."
    ),
}


SYSTEM_PROMPT = f"""
You are Link3's Sales Funnel AI Closer for Link3 ISP sales.
You are running on a small local LLM, so be concise, structured, and practical.

Available B2B packages:
{", ".join(PACKAGES["B2B"])}

Available B2C packages:
{", ".join(PACKAGES["B2C"])}

Link3 sales context to consider:
- Common service areas: {", ".join(LINK3_MARKET_CONTEXT["areas"])}
- Common customer types: {", ".join(LINK3_MARKET_CONTEXT["customer_types"])}
- Payment channels: {", ".join(LINK3_MARKET_CONTEXT["payment_channels"])}
- Onboarding documents: {", ".join(LINK3_MARKET_CONTEXT["documents"])}
- Common sales objections: {", ".join(LINK3_MARKET_CONTEXT["local_objections"])}

Analyze the lead message and return ONLY one valid JSON object.

Important decision rules:
- B2B signals: company, office, employees, branch, dedicated, SLA, static IP, procurement, quotation.
- Link3 B2B signals: office area, branch, trade license, corporate bill, SLA, public IP, MikroTik, CCTV, Zoom, backup link.
- B2C signals: home, family, gaming, streaming, student, budget, apartment, freelancer, YouTube, Facebook, Netflix.
- Link3 B2C signals: bKash/Nagad payment, installation charge, night speed, area coverage, gaming ping, work-from-home calls.
- High buying intent: urgent timeline, pricing request, site survey, competitor comparison, decision deadline.
- Retention: existing Link3 customer asking upgrade, backup, complaint, or renewal.
- Lead score should reflect conversion probability, not politeness.
- If information is missing, say what sales should ask next.
- Suggested reply must be ready to send by WhatsApp or email in natural Link3 sales style.
- Use BDT when talking about pricing. Do not invent exact prices unless the lead gave a number.
- Mention area coverage check, installation feasibility, and site survey when relevant.
- For B2B, mention corporate quotation, SLA, static public IP, backup link, and required documents when relevant.
- For B2C, mention home package fit, installation charge question, payment channel, speed needs, and support.
- Use mostly English with light Bangla where helpful if the customer used Bangla.
- The suggested_reply must directly reflect the same lead context: area, requirement, pain point, package fit, urgency, installation/site survey, pricing/payment, and next step.
- The objection_response must answer the most likely objection from this exact lead, such as price, installation charge, coverage, night speed, public IP cost, contract, or support.
- The crm_note must summarize the same context for a salesperson: segment, area, package need, buying signals, objections, next action, and follow-up priority.

Return exactly this JSON shape:
{{
  "b2b_b2c": "B2B or B2C",
  "funnel_stage": "Awareness / Interest / Consideration / Intent / Purchase / Retention",
  "deal_temperature": "Hot / Warm / Cold",
  "lead_score": 0,
  "urgency": "Low / Med / High",
  "confidence": 0,
  "estimated_deal_size": "Small / Medium / Large / Unknown",
  "location_area": "detected service area or Unknown",
  "customer_type": "home user / freelancer / SME / corporate / shop / school / clinic / unknown",
  "package_recommendation": "one best package or bundle",
  "bdt_pricing_note": "what pricing should be confirmed in BDT",
  "payment_preference": "bKash / Nagad / bank transfer / card / cash collection / Unknown",
  "installation_readiness": "Ready / Need area check / Need site survey / Unknown",
  "required_documents": ["NID or trade license if relevant"],
  "buying_signals": ["short signal 1", "short signal 2"],
  "pain_points": ["short pain point"],
  "objections": ["possible objection or missing info"],
  "sales_context_notes": ["Link3-specific sales observation"],
  "competitor_risk": false,
  "decision_maker_status": "Likely / Unknown / Not decision maker",
  "next_action": "single best next action",
  "follow_up_plan": [
    {{"step": 1, "timing": "Now", "action": "what to do"}},
    {{"step": 2, "timing": "Within 2 hours", "action": "what to do"}},
    {{"step": 3, "timing": "Tomorrow", "action": "what to do"}}
  ],
  "suggested_reply": "ready-to-send message",
  "objection_response": "short response if the customer hesitates",
  "crm_note": "short CRM summary",
  "reason": "one sentence explaining the decision"
}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_model() -> str:
    try:
        resp = requests.get(MODELS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            model_id = data["data"][0].get("id", DEFAULT_MODEL)
            return model_id.split("/")[-1] if "/" in model_id else model_id
    except Exception:
        pass
    return DEFAULT_MODEL


def extract_json(content: str) -> Dict[str, Any]:
    clean = content.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group())


def clamp_score(value: Any, default: int = 50) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except Exception:
        return default


def as_list(value: Any, fallback: Optional[List[str]] = None) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback or []


def detect_area(text: str) -> str:
    text_lower = text.lower()
    for area in LINK3_MARKET_CONTEXT["areas"]:
        if area.lower() in text_lower:
            return area
    return "Unknown"


def detect_payment_channel(text: str) -> str:
    text_lower = text.lower()
    for channel in LINK3_MARKET_CONTEXT["payment_channels"]:
        if channel.lower() in text_lower:
            return channel
    return "Unknown"


def normalize_ai_result(result: Dict[str, Any]) -> Dict[str, Any]:
    segment = str(result.get("b2b_b2c", "B2C")).upper()
    if "B2B" not in segment:
        segment = "B2C"
    else:
        segment = "B2B"

    stage = str(result.get("funnel_stage", "Interest"))
    if stage not in FUNNEL_STAGES:
        stage = "Interest"

    temp = str(result.get("deal_temperature", "Warm")).title()
    if temp not in {"Hot", "Warm", "Cold"}:
        temp = "Warm"

    urgency = str(result.get("urgency", "Med")).title()
    if urgency not in {"Low", "Med", "High"}:
        urgency = "Med"

    follow_up = result.get("follow_up_plan", [])
    if not isinstance(follow_up, list) or not follow_up:
        follow_up = [
            {"step": 1, "timing": "Now", "action": result.get("next_action", "Follow up with the lead")},
            {"step": 2, "timing": "Within 2 hours", "action": "Confirm requirement, location, and package fit"},
            {"step": 3, "timing": "Tomorrow", "action": "Send reminder with quotation or plan details"},
        ]

    location_area = result.get("location_area", "Unknown")
    customer_type = result.get("customer_type", "unknown")
    package_recommendation = result.get("package_recommendation", "Best-fit internet package")
    pricing_note = result.get("bdt_pricing_note", "Confirm monthly bill, installation charge, and VAT in BDT.")
    installation_readiness = result.get("installation_readiness", "Unknown")
    next_action = result.get("next_action", NEXT_ACTIONS.get(stage, "Follow up"))
    buying_signals = as_list(result.get("buying_signals"), ["Lead asked about internet service"])
    pain_points = as_list(result.get("pain_points"), ["Need better connectivity"])
    objections = as_list(result.get("objections"), ["Budget, timeline, or decision maker not confirmed"])
    sales_context_notes = as_list(result.get("sales_context_notes"), ["Link3 sales context applied"])

    contextual_reply = (
        f"Thanks for reaching out. Based on your requirement"
        f"{' in ' + location_area if location_area != 'Unknown' else ''}, "
        f"{package_recommendation} looks like a good fit. We can confirm availability, "
        f"installation feasibility, and pricing details before the next step. {next_action}"
    )
    contextual_objection = (
        f"If pricing, installation charge, or coverage is a concern, we can verify the exact area, "
        f"usage need, and package fit first, then share a clear quotation. Current note: {pricing_note}"
    )
    contextual_crm_note = (
        f"{temp} {segment} lead"
        f"{' in ' + location_area if location_area != 'Unknown' else ''}; "
        f"customer type: {customer_type}; package fit: {package_recommendation}; "
        f"signals: {', '.join(buying_signals[:3])}; pain points: {', '.join(pain_points[:2])}; "
        f"objections/missing info: {', '.join(objections[:2])}; installation: {installation_readiness}; "
        f"next action: {next_action}."
    )

    suggested_reply = result.get("suggested_reply", "")
    if len(str(suggested_reply).strip()) < 40:
        suggested_reply = contextual_reply

    objection_response = result.get("objection_response", "")
    if len(str(objection_response).strip()) < 30:
        objection_response = contextual_objection

    crm_note = result.get("crm_note", "")
    if len(str(crm_note).strip()) < 40:
        crm_note = contextual_crm_note

    return {
        "b2b_b2c": segment,
        "funnel_stage": stage,
        "deal_temperature": temp,
        "lead_score": clamp_score(result.get("lead_score"), 50),
        "urgency": urgency,
        "confidence": clamp_score(result.get("confidence"), 70),
        "estimated_deal_size": result.get("estimated_deal_size", "Unknown"),
        "location_area": location_area,
        "customer_type": customer_type,
        "package_recommendation": package_recommendation,
        "bdt_pricing_note": pricing_note,
        "payment_preference": result.get("payment_preference", "Unknown"),
        "installation_readiness": installation_readiness,
        "required_documents": as_list(result.get("required_documents"), ["Confirm NID or trade license based on customer type"]),
        "buying_signals": buying_signals,
        "pain_points": pain_points,
        "objections": objections,
        "sales_context_notes": sales_context_notes,
        "competitor_risk": bool(result.get("competitor_risk", False)),
        "decision_maker_status": result.get("decision_maker_status", "Unknown"),
        "next_action": next_action,
        "follow_up_plan": follow_up[:3],
        "suggested_reply": suggested_reply,
        "objection_response": objection_response,
        "crm_note": crm_note,
        "reason": result.get("reason", "AI assessed the lead using segment, intent, urgency, and buying signals."),
    }


def rule_based_score(lead_text: str) -> Dict[str, Any]:
    text = lead_text.lower()
    score = 20
    matched_rules: List[str] = []

    b2b_terms = [
        "office",
        "company",
        "employee",
        "employees",
        "branch",
        "corporate",
        "business",
        "dedicated",
        "sla",
        "trade license",
        "static ip",
        "public ip",
        "mikrotik",
        "cctv",
    ]
    b2c_terms = [
        "home",
        "family",
        "gaming",
        "student",
        "apartment",
        "budget",
        "freelancer",
        "বাসা",
        "ভাই",
        "youtube",
        "facebook",
        "netflix",
        "valorant",
    ]
    urgent_terms = ["urgent", "today", "tomorrow", "this week", "deadline", "asap", "quick"]
    no_urgency_terms = ["no urgent", "not urgent", "no hurry", "not in a hurry"]
    competitor_terms = ["competitor", "another provider", "comparing", "other isp"]
    intent_terms = [
        "quotation",
        "pricing",
        "price",
        "site survey",
        "install",
        "installation",
        "dedicated fiber",
        "static ip",
        "public ip",
        "bdt",
        "monthly",
        "charge",
        "কত",
        "চাই",
    ]
    retention_terms = ["already use", "existing", "upgrade", "renewal", "backup"]
    local_payment_terms = ["bkash", "বিকাশ", "nagad", "নগদ", "rocket", "bank transfer", "cash"]
    local_area = detect_area(lead_text)
    payment_channel = detect_payment_channel(lead_text)

    segment = "B2C"
    if any(term in text for term in b2b_terms):
        segment = "B2B"
        score += 20
        matched_rules.append("B2B business keywords")
    elif any(term in text for term in b2c_terms):
        matched_rules.append("B2C home-user keywords")

    if any(term in text for term in urgent_terms) and not any(term in text for term in no_urgency_terms):
        score += 20
        matched_rules.append("Urgent timeline")
    if any(term in text for term in competitor_terms):
        score += 15
        matched_rules.append("Competitor risk")
    if any(term in text for term in intent_terms):
        score += 20
        matched_rules.append("Strong buying intent")
    if any(term in text for term in retention_terms):
        score += 15
        matched_rules.append("Existing customer or upgrade signal")
    if local_area != "Unknown":
        score += 8
        matched_rules.append(f"Service area detected: {local_area}")
    if any(term in text for term in local_payment_terms):
        score += 5
        matched_rules.append("Local payment channel mentioned")
    if any(term in text for term in ["installation charge", "area coverage", "coverage", "site survey"]):
        score += 8
        matched_rules.append("Installation or coverage concern")

    employee_match = re.search(r"(\d+)\s*(employee|employees|staff|people|users)", text)
    employee_count = int(employee_match.group(1)) if employee_match else None
    if employee_count:
        if employee_count >= 50:
            score += 15
            matched_rules.append("Large team size")
        elif employee_count >= 20:
            score += 10
            matched_rules.append("Medium team size")

    score = clamp_score(score)
    if score >= 80:
        temperature = "Hot"
        priority = "Senior sales follow-up"
    elif score >= 55:
        temperature = "Warm"
        priority = "Sales follow-up"
    else:
        temperature = "Cold"
        priority = "Nurture sequence"

    return {
        "segment": segment,
        "score": score,
        "temperature": temperature,
        "priority": priority,
        "detected_area": local_area,
        "payment_channel": payment_channel,
        "matched_rules": matched_rules or ["No strong deterministic rule matched"],
    }


def call_sales_ai(lead_text: str, model_name: str) -> Tuple[Dict[str, Any], str, float]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": lead_text},
        ],
        "temperature": 0.25,
        "max_tokens": 900,
    }

    started_at = time.perf_counter()
    response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    latency = time.perf_counter() - started_at
    return normalize_ai_result(extract_json(raw_content)), raw_content, latency


def reset_dashboard() -> None:
    st.session_state.sales_result = None
    st.session_state.raw_ai_response = ""
    st.session_state.rule_result = None
    st.session_state.latency = 0.0


def render_list(items: List[str]) -> None:
    for item in items:
        st.write(f"- {item}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Link3 Sales AI Closer v3", page_icon=":chart_with_upwards_trend:", layout="wide")

if "sales_result" not in st.session_state:
    st.session_state.sales_result = None
if "raw_ai_response" not in st.session_state:
    st.session_state.raw_ai_response = ""
if "rule_result" not in st.session_state:
    st.session_state.rule_result = None
if "latency" not in st.session_state:
    st.session_state.latency = 0.0
if "current_model" not in st.session_state:
    st.session_state.current_model = "Unknown"
if "lead_input" not in st.session_state:
    st.session_state.lead_input = DEMO_LEADS["Hot B2B Banani lead"]

st.title("Link3 Sales Funnel AI Closer v3")
st.markdown(
    "**Marketing & Sales Team** - AI lead scoring, area context, pricing notes, "
    "package fit, objection handling, and follow-up planning"
)
st.markdown("---")

with st.sidebar:
    st.subheader("Demo Leads")
    selected_demo = st.selectbox("Choose a demo", list(DEMO_LEADS.keys()))
    if st.button("Load Demo Lead", use_container_width=True):
        st.session_state.lead_input = DEMO_LEADS[selected_demo]
        reset_dashboard()

    st.divider()
    st.subheader("Local Model")
    if st.session_state.current_model == "Unknown":
        st.session_state.current_model = get_current_model()
    st.code(st.session_state.current_model)

left, right = st.columns([1, 1.35])

with left:
    st.subheader("Lead Inquiry")
    lead_input = st.text_area(
        "Paste the lead message from WhatsApp, email, Facebook, call note, or CRM",
        key="lead_input",
        height=240,
    )

    c1, c2 = st.columns(2)
    with c1:
        analyze_btn = st.button("Analyze Lead", use_container_width=True, type="primary")
    with c2:
        st.button("Clear Result", on_click=reset_dashboard, use_container_width=True)

    st.markdown("**What v3 asks the LLM to decide**")
    st.write("- Lead score and deal temperature")
    st.write("- Service area, customer type, and payment context")
    st.write("- Buying signals, pain points, and objections")
    st.write("- ISP package recommendation with pricing note")
    st.write("- Installation readiness and required documents")
    st.write("- 3-step follow-up plan")
    st.write("- CRM-ready summary")

if analyze_btn and lead_input.strip():
    current_model_name = get_current_model()
    st.session_state.current_model = current_model_name
    st.session_state.rule_result = rule_based_score(lead_input)

    try:
        with st.spinner(f"Local AI model ({current_model_name}) is making the sales decision..."):
            result, raw_content, latency = call_sales_ai(lead_input, current_model_name)
            st.session_state.sales_result = result
            st.session_state.raw_ai_response = raw_content
            st.session_state.latency = latency
    except Exception as exc:
        st.session_state.sales_result = None
        st.session_state.raw_ai_response = ""
        st.session_state.latency = 0.0
        st.error(f"Could not get AI decision from LM Studio: {exc}")
        st.info("The rule-based score still works. Start LM Studio on localhost:1234 to show AI reasoning.")

with right:
    st.subheader("AI Deal Recommendation")

    res = st.session_state.sales_result
    rules = st.session_state.rule_result

    if res:
        metric_cols = st.columns(4)
        metric_cols[0].metric("AI Lead Score", f"{res['lead_score']}/100")
        metric_cols[1].metric("Deal Temp", res["deal_temperature"])
        metric_cols[2].metric("Funnel Stage", res["funnel_stage"])
        metric_cols[3].metric("Urgency", res["urgency"])

        if res["deal_temperature"] == "Hot" or res["urgency"] == "High":
            st.error("High priority lead - contact fast")
        elif res["deal_temperature"] == "Warm":
            st.warning("Warm lead - nurture with a clear offer")
        else:
            st.info("Cold lead - send educational content")

        st.write(f"**Segment:** {res['b2b_b2c']}")
        st.write(f"**Location / Area:** {res['location_area']}")
        st.write(f"**Customer type:** {res['customer_type']}")
        st.write(f"**Estimated deal size:** {res['estimated_deal_size']}")
        st.write(f"**Recommended package:** {res['package_recommendation']}")
        st.write(f"**Pricing note:** {res['bdt_pricing_note']}")
        st.write(f"**Next action:** {res['next_action']}")

        st.progress(res["confidence"] / 100, text=f"AI Confidence: {res['confidence']}%")
        st.caption(f"Reason: {res['reason']}")

        st.divider()
        detail_tabs = st.tabs(["Sales Context", "Signals", "Follow-up", "Reply", "CRM", "Rule Check", "Raw JSON"])

        with detail_tabs[0]:
            a, b, c = st.columns(3)
            with a:
                st.markdown("**Payment Preference**")
                st.info(res["payment_preference"])
            with b:
                st.markdown("**Installation Readiness**")
                st.info(res["installation_readiness"])
            with c:
                st.markdown("**Required Documents**")
                render_list(res["required_documents"])

            st.markdown("**Sales Context Notes**")
            render_list(res["sales_context_notes"])

        with detail_tabs[1]:
            a, b, c = st.columns(3)
            with a:
                st.markdown("**Buying Signals**")
                render_list(res["buying_signals"])
            with b:
                st.markdown("**Pain Points**")
                render_list(res["pain_points"])
            with c:
                st.markdown("**Objections / Missing Info**")
                render_list(res["objections"])

            st.write(f"**Competitor risk:** {'Yes' if res['competitor_risk'] else 'No'}")
            st.write(f"**Decision maker status:** {res['decision_maker_status']}")

        with detail_tabs[2]:
            for item in res["follow_up_plan"]:
                step = item.get("step", "-") if isinstance(item, dict) else "-"
                timing = item.get("timing", "Next") if isinstance(item, dict) else "Next"
                action = item.get("action", str(item)) if isinstance(item, dict) else str(item)
                st.write(f"**Step {step} - {timing}:** {action}")

        with detail_tabs[3]:
            st.markdown("**Ready-to-send reply**")
            st.info(res["suggested_reply"])
            st.markdown("**Objection response**")
            st.warning(res["objection_response"])

        with detail_tabs[4]:
            st.markdown("**CRM-ready note**")
            st.code(res["crm_note"])

        with detail_tabs[5]:
            if rules:
                c1, c2, c3 = st.columns(3)
                c1.metric("Rule Score", f"{rules['score']}/100")
                c2.metric("Rule Segment", rules["segment"])
                c3.metric("Rule Priority", rules["priority"])
                st.write(f"**Rule detected area:** {rules.get('detected_area', 'Unknown')}")
                st.write(f"**Rule payment channel:** {rules.get('payment_channel', 'Unknown')}")
                st.markdown("**Matched deterministic rules**")
                render_list(rules["matched_rules"])

                st.caption(
                    "Teaching point: the rule score is transparent but shallow; the LLM adds package fit, "
                    "objection handling, CRM summary, and next-step planning."
                )

        with detail_tabs[6]:
            st.code(json.dumps(res, indent=2), language="json")

    else:
        st.info("Paste or load a lead, then click Analyze Lead.")

st.divider()
bottom_cols = st.columns(4)
bottom_cols[0].metric("Active Model", st.session_state.current_model)
bottom_cols[1].metric("AI Latency", f"{st.session_state.latency:.2f}s")
bottom_cols[2].metric("Script Version", "v3")
bottom_cols[3].metric("Decision Mode", "AI + Rule Check")
