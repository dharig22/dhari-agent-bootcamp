"""
Vector One: Agent Experiment 2 (Python Orchestrator + Worker 0 Extractor + Worker 1 Scorer + Worker 2 Auditor)

What changed (per your Slack discussion):
- Worker 0 is the ONLY worker that sees the full application packet.
  -> Worker 0 extracts ONLY the relevant evidence + gaps per criterion.
- Worker 1 does NOT get the base use case / full packet anymore.
  -> Worker 1 scores purely from Worker 0’s extracted evidence pack + rubric.
- Worker 2 is a true auditor (less passive):
  -> Worker 2 challenges Worker 1’s scores based on evidence sufficiency + gaps,
     and must explicitly decide if scores are "too high / fair / too low".
- Orchestrator is the only component that sees: Worker0 + Worker1 + Worker2 + rubric + raw packet.

Run:
  python experiment-two.py
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import agents
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage

from src.utils.client_manager import AsyncClientManager
from src.utils.gradio import COMMON_GRADIO_CONFIG
from src.utils.agent_session import get_or_create_session

# ----------------------------
# Configuration
# ----------------------------

DATASET_CSV_PATH = "Vector-one/data/use_case_clean.json"
RUBRIC_CSV_PATH = "Vector-one/data/evaluation_rubric_machine_readable.json"

client_manager = AsyncClientManager()
DEFAULT_MODEL = client_manager.configs.default_planner_model


# ----------------------------
# Utilities: loading + normalization
# ----------------------------

def load_applications_json(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_rubric_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _clean_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def infer_company_identifier_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    candidates: List[str] = []
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("unnamed"):
            continue
        if any(k in cl for k in ["company", "organization", "org", "applicant", "name"]):
            candidates.append(c)
    primary = candidates[0] if candidates else None
    return primary, candidates

SECTION_ANCHORS = [
    ("project_summary", r"Project Summary:"),
    ("deliverables", r"What specific outputs or deliverables do you envision for this project"),
    ("success_metrics", r"How will you measure success for this project"),
    ("dataset_description", r"Please describe the dataset"),
    ("dataset_readiness", r"What is the current state of readiness of this dataset"),
    ("dataset_pii", r"Does the dataset contain any personally identifiable information"),
    ("infrastructure", r"What infrastructure or platforms do you plan to use"),
    ("genai_llms", r"Have you already explored.*(Generative AI|LLMs)"),
    ("team", r"Please describe your technical team"),
    ("additional_context", r"Is there any additional context"),
    ("ideal_mla", r"In one or two sentences describe the ideal MLA candidate"),
]

def _extract_sections_from_qa(qa_text: str) -> Dict[str, str]:
    if not isinstance(qa_text, str) or not qa_text.strip():
        return {}

    hits = []
    for key, pattern in SECTION_ANCHORS:
        m = re.search(pattern, qa_text, flags=re.IGNORECASE)
        if m:
            hits.append((m.start(), m.end(), key))

    hits.sort(key=lambda x: x[0])
    if not hits:
        return {}

    sections: Dict[str, str] = {}
    for i, (start, end, key) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(qa_text)
        chunk = qa_text[end:next_start].strip()
        chunk = re.sub(r"\n{3,}", "\n\n", chunk).strip()
        if chunk:
            sections[key] = chunk[:6000]
    return sections

def application_packet_from_row(df: pd.DataFrame, row_idx: int) -> Dict[str, Any]:
    row = df.iloc[row_idx].to_dict()

    cleaned: Dict[str, Any] = {}
    for k, v in row.items():
        ck = _clean_key(str(k))
        if v is None:
            cleaned[ck] = None
        elif isinstance(v, float) and pd.isna(v):
            cleaned[ck] = None
        else:
            cleaned[ck] = v

    qa = cleaned.get("qa_consolidated")
    if isinstance(qa, str) and qa.strip():
        derived = _extract_sections_from_qa(qa)
        for k, txt in derived.items():
            if k not in cleaned:
                cleaned[k] = txt

    primary_id_col, _ = infer_company_identifier_columns(df)
    company_id = None
    if primary_id_col:
        company_id = row.get(primary_id_col)
    if not company_id or (isinstance(company_id, float) and pd.isna(company_id)):
        company_id = f"row_{row_idx + 1}"

    return {
        "company_id": str(company_id),
        "row_index_1_based": row_idx + 1,
        "application_original_headers": row,
        "application": cleaned,
    }

def resolve_application_selector(df: pd.DataFrame, selector: str) -> int:
    s = selector.strip()

    m = re.search(r"\brow\s+(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n < 1 or n > len(df):
            raise ValueError(f"Row {n} out of range (1..{len(df)})")
        return n - 1

    m = re.search(r"\bcompany\s+(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n < 1 or n > len(df):
            raise ValueError(f"Company {n} out of range (1..{len(df)})")
        return n - 1

    if re.fullmatch(r"\d+", s):
        n = int(s)
        if n < 1 or n > len(df):
            raise ValueError(f"Row {n} out of range (1..{len(df)})")
        return n - 1

    primary_id_col, candidate_cols = infer_company_identifier_columns(df)
    cols_to_try: List[str] = []
    if primary_id_col:
        cols_to_try.append(primary_id_col)
    cols_to_try += [c for c in candidate_cols if c not in cols_to_try]

    for col in cols_to_try:
        series = df[col].astype(str)
        matches = series[series.str.strip() == s]
        if len(matches) == 1:
            return int(matches.index[0])
        matches = series[series.str.contains(re.escape(s), case=False, na=False)]
        if len(matches) == 1:
            return int(matches.index[0])

    raise ValueError("Could not resolve selector. Try 'company 7' or 'row 12'.")

def _safe_json_loads(s: str) -> Any:
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(s[start:end + 1])

def _get_level_bands(rubric_json: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
    bands = {}
    for lvl in rubric_json.get("levels", []):
        level = int(lvl["level"])
        mn = int(round(float(lvl["score_range"]["min"])))
        mx = int(round(float(lvl["score_range"]["max"])))
        bands[level] = (mn, mx)
    return bands

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        xi = int(x)
    except Exception:
        return default
    return max(lo, min(hi, xi))

def _validate_level_percent(final_scores: Dict[str, Any], rubric_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    bands = _get_level_bands(rubric_json)
    fixes = []

    for crit, obj in final_scores.items():
        if not isinstance(obj, dict):
            continue
        level = _clamp_int(obj.get("level"), 0, 4, default=0)
        lo, hi = bands.get(level, (0, 100))
        mid = int(round((lo + hi) / 2))
        percent = _clamp_int(obj.get("percent"), 0, 100, default=mid)

        if percent < lo or percent > hi:
            old = percent
            percent = max(lo, min(hi, percent))
            fixes.append({"criterion": crit, "issue": "percent_outside_band", "from": old, "to": percent, "band": [lo, hi]})

        obj["level"] = level
        obj["percent"] = percent

    return fixes

def _compute_average_percent(final_scores: Dict[str, Any]) -> float:
    vals: List[float] = []
    for v in final_scores.values():
        if isinstance(v, dict) and "percent" in v:
            try:
                vals.append(float(v["percent"]))
            except Exception:
                pass
    return round(sum(vals) / len(vals), 1) if vals else 0.0

def _derive_overall_level_from_percent(overall_percent: float, rubric_json: Dict[str, Any]) -> int:
    bands = _get_level_bands(rubric_json)
    for level in sorted(bands.keys()):
        lo, hi = bands[level]
        if lo <= overall_percent <= hi:
            return int(level)
    return 0 if overall_percent < 0 else 4

def _derive_confidence_from_gaps(scores: Dict[str, Any]) -> str:
    total_gaps = 0
    criteria_with_gaps = 0
    for v in scores.values():
        if isinstance(v, dict):
            gaps = v.get("gaps") or []
            if isinstance(gaps, list) and len(gaps) > 0:
                criteria_with_gaps += 1
                total_gaps += len(gaps)
    if criteria_with_gaps == 0:
        return "high"
    if criteria_with_gaps <= 2 and total_gaps <= 3:
        return "medium"
    return "low"


# ----------------------------
# Local store
# ----------------------------

@dataclass
class LocalDataStore:
    applications_df: pd.DataFrame
    rubric: Dict[str, Any]

    def list_applications(self, limit: int = 20) -> Dict[str, Any]:
        primary_id_col, _ = infer_company_identifier_columns(self.applications_df)
        items = []
        for i in range(min(limit, len(self.applications_df))):
            cid = str(self.applications_df.iloc[i][primary_id_col]) if primary_id_col else f"row_{i+1}"
            items.append({"row": i + 1, "company_id": cid})
        return {"count": len(self.applications_df), "primary_id_col": primary_id_col, "items": items}

    def get_application(self, selector: str) -> Dict[str, Any]:
        idx0 = resolve_application_selector(self.applications_df, selector)
        return application_packet_from_row(self.applications_df, idx0)

    def get_rubric(self) -> Dict[str, Any]:
        return {"rubric_sheets": self.rubric}


# ----------------------------
# Worker agents (LLM)
# ----------------------------

WORKER_0_INSTRUCTIONS = """
You are Worker 0 (Evidence Extractor / Summarizer).

You will receive:
- application_packet (structured fields)
- rubric JSON (rubric_sheets)

Goal:
Produce an "evidence pack" per criterion so downstream scorers do NOT need the full application.

Rules:
- Do NOT score. Do NOT assign levels or percents.
- Do NOT invent details.
- Use the MOST SPECIFIC derived fields when possible (e.g., dataset_description, infrastructure, team).
- Only use qa_consolidated if the needed evidence is not available in derived fields.
- Provide concise quotes (10–25 words) that clearly support evidence.
- Identify gaps / unknown dependencies per criterion.

Return JSON exactly in this shape:

{
  "company_id": "...",
  "evidence_pack": {
    "use_case_clarity_alignment": {
      "evidence_fields": ["project_summary", "deliverables", "success_metrics"],
      "evidence_quotes": ["...10–25 words...", "..."],
      "summary": "2–4 sentences: what the use case is and who it serves",
      "gaps": ["..."]
    },
    "dataset_accessibility_suitability": {
      "evidence_fields": ["dataset_description", "dataset_readiness", "dataset_pii"],
      "evidence_quotes": ["..."],
      "summary": "2–4 sentences: what data exists + accessibility constraints",
      "gaps": ["..."]
    },
    "infrastructure_tooling_readiness": {
      "evidence_fields": ["infrastructure", "genai_llms", "team"],
      "evidence_quotes": ["..."],
      "summary": "2–4 sentences: environment/tooling status",
      "gaps": ["..."]
    },
    "execution_feasibility": {
      "evidence_fields": ["team", "deliverables", "additional_context"],
      "evidence_quotes": ["..."],
      "summary": "2–4 sentences: feasibility signals (team, scope, timeline)",
      "gaps": ["..."]
    }
  },
  "global_red_flags": ["..."],
  "missing_key_fields": ["..."]
}
"""

WORKER_1_INSTRUCTIONS = """
You are Worker 1 (Scorer).

You will receive:
- rubric JSON (rubric_sheets)
- Worker 0 evidence_pack ONLY (you will NOT receive full application)

Your job:
Score EXACTLY these 4 criteria:
- use_case_clarity_alignment
- dataset_accessibility_suitability
- infrastructure_tooling_readiness
- execution_feasibility

Scoring output per criterion:
- level: integer 0–4
- percent: integer 0–100 AND MUST be inside the official band for that level:
  rubric_sheets.levels[level].score_range.min/max

How to score:
- Choose BEST-matching level by comparing evidence_pack to:
  rubric_sheets.rubric[*].readiness_by_level[level]
- Place percent within level band:
  * bottom of band if barely meets
  * mid-band if clearly meets
  * top of band only if gaps is empty or minor

Hard rule:
- If evidence_pack.gaps contains any MATERIAL unknown dependency for that criterion,
  keep percent in the lower or mid part of the band (NOT top).

Return JSON exactly with this shape:

{
  "company_id": "...",
  "scores": {
    "use_case_clarity_alignment": {
      "level": 0,
      "percent": 0,
      "rationale": "...",
      "percent_rationale": "...",
      "evidence_fields": ["..."],
      "evidence_quotes": ["..."],
      "gaps": ["..."]
    },
    "dataset_accessibility_suitability": { ... },
    "infrastructure_tooling_readiness": { ... },
    "execution_feasibility": { ... }
  },
  "overall": {
    "recommendation": "Proceed | Proceed with conditions | Hold | Reject",
    "top_risks": ["..."],
    "priority_followups": ["..."]
  }
}
"""

WORKER_2_INSTRUCTIONS = """
You are Worker 2 (Auditor / Verifier).

You will receive:
- rubric JSON (rubric_sheets)
- Worker 0 evidence_pack
- Worker 1 output (scores + rationale)

Your job:
Audit whether Worker 1's scores are FAIR given the evidence sufficiency + gaps.

You must do ALL of this:
1) Decide if each criterion score is: "too_high | fair | too_low"
2) If too_high or too_low, adjust level and/or percent (percent must stay inside level band)
3) Explicitly cite why evidence/gaps justify the change (or fairness)

Bias:
- Prefer conservative scoring when evidence is thin or gaps are material.
- If Worker 0 gaps include a material unknown dependency, the score cannot be top-of-band.

Return JSON exactly with this shape:

{
  "company_id": "...",
  "final_scores": {
    "use_case_clarity_alignment": {
      "level": 0,
      "percent": 0,
      "judgement": "too_high | fair | too_low",
      "decision": "accept | adjust",
      "reason": "...",
      "evidence_fields": ["..."],
      "evidence_quotes": ["..."],
      "gaps": ["..."]
    },
    "dataset_accessibility_suitability": { ... },
    "infrastructure_tooling_readiness": { ... },
    "execution_feasibility": { ... }
  },
  "adjustments": [
    {"criterion": "dataset_accessibility_suitability", "from_level": 4, "to_level": 4, "from_percent": 98, "to_percent": 92, "reason": "..."}
  ],
  "confidence": "low | medium | high",
  "notes": "short overall notes"
}
"""

def build_workers(model_name: str, openai_client):
    worker0 = agents.Agent(
        name="Worker0_Extractor",
        instructions=WORKER_0_INSTRUCTIONS,
        tools=[],
        model=agents.OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client),
    )
    worker1 = agents.Agent(
        name="Worker1_Scorer",
        instructions=WORKER_1_INSTRUCTIONS,
        tools=[],
        model=agents.OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client),
    )
    worker2 = agents.Agent(
        name="Worker2_Auditor",
        instructions=WORKER_2_INSTRUCTIONS,
        tools=[],
        model=agents.OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client),
    )
    return worker0, worker1, worker2


# Globals set in __main__
store: LocalDataStore
worker0_agent: agents.Agent
worker1_agent: agents.Agent
worker2_agent: agents.Agent


# ----------------------------
# Gradio handler (Python orchestrator)
# ----------------------------

async def _main(
    query: str, history: List[ChatMessage], session_state: Dict[str, Any]
) -> AsyncGenerator[List[ChatMessage], Any]:
    turn_messages: List[ChatMessage] = []
    session = get_or_create_session(history, session_state)

    try:
        q = (query or "").strip()

        # LIST
        if re.search(r"\blist\b", q, flags=re.IGNORECASE):
            listed = store.list_applications(limit=20)
            lines = [f"Here are the first {len(listed['items'])} available applications:"]
            for it in listed["items"]:
                lines.append(f"- {it['company_id']} (row {it['row']})")
            turn_messages.append(ChatMessage(role="assistant", content="\n".join(lines)))
            yield turn_messages
            return

        # EVALUATE
        selector = re.sub(r"^\s*(evaluate|score)\s+", "", q, flags=re.IGNORECASE).strip()

        packet = store.get_application(selector)
        rubric_obj = store.get_rubric()

        # -------------------------
        # Worker 0: evidence extract (ONLY worker that sees full application)
        # -------------------------
        w0_payload = {
            "application_packet": packet,
            "rubric": rubric_obj,
        }
        w0_result = await agents.Runner.run(
            worker0_agent,
            input=json.dumps(w0_payload, ensure_ascii=False),
            session=session,
            max_turns=10,
        )
        w0_json = _safe_json_loads(str(w0_result.final_output))
        evidence_pack = w0_json.get("evidence_pack", {})

        # -------------------------
        # Worker 1: score based ONLY on evidence_pack + rubric
        # -------------------------
        w1_payload = {
            "company_id": packet["company_id"],
            "rubric": rubric_obj,
            "evidence_pack": evidence_pack,
            # explicitly NOT sending application_packet
        }
        w1_result = await agents.Runner.run(
            worker1_agent,
            input=json.dumps(w1_payload, ensure_ascii=False),
            session=session,
            max_turns=10,
        )
        w1_json = _safe_json_loads(str(w1_result.final_output))

        # -------------------------
        # Worker 2: audit Worker 1 using evidence_pack + rubric
        # -------------------------
        w2_payload = {
            "company_id": packet["company_id"],
            "rubric": rubric_obj,
            "evidence_pack": evidence_pack,
            "worker1_output": w1_json,
        }
        w2_result = await agents.Runner.run(
            worker2_agent,
            input=json.dumps(w2_payload, ensure_ascii=False),
            session=session,
            max_turns=10,
        )
        w2_json = _safe_json_loads(str(w2_result.final_output))

        final_scores = w2_json.get("final_scores", {})
        percent_fixes = _validate_level_percent(final_scores, store.rubric)

        overall_percent = _compute_average_percent(final_scores)
        overall_level = _derive_overall_level_from_percent(overall_percent, store.rubric)

        # Recommendation source:
        # - keep Worker1's overall recommendation (it “scores”)
        # - but Worker2 confidence is the “audit” confidence
        recommendation = w1_json.get("overall", {}).get("recommendation", "Proceed with conditions")
        top_risks = w1_json.get("overall", {}).get("top_risks", [])
        priority_followups = w1_json.get("overall", {}).get("priority_followups", [])

        confidence = w2_json.get("confidence") or _derive_confidence_from_gaps(final_scores)

        consolidated = {
            "company_id": packet["company_id"],
            "scores": final_scores,
            "overall": {
                "overall_level": overall_level,
                "overall_percent": overall_percent,
                "recommendation": recommendation,
                "confidence": confidence,
                "top_risks": top_risks,
                "priority_followups": priority_followups,
                "global_red_flags": w0_json.get("global_red_flags", []),
            },
            "audit": {
                "worker0_evidence": w0_json,
                "worker1_scoring": w1_json,
                "worker2_audit": w2_json,
                "adjustments": w2_json.get("adjustments", []),
                "percent_fixes": percent_fixes,
            },
        }

        summary_lines = [
            f"Evaluation for **{packet['company_id']}** (row {packet['row_index_1_based']}):",
            f"- Overall level: **{overall_level}/4**",
            f"- Overall percent: **{overall_percent}%**",
            f"- Recommendation: **{recommendation}**",
            f"- Confidence: **{confidence}**",
        ]

        for crit, v in final_scores.items():
            if isinstance(v, dict):
                summary_lines.append(f"- {crit}: Level **{v.get('level')}**, **{v.get('percent')}%** ({v.get('judgement','')})")
                gaps = v.get("gaps") or []
                if gaps:
                    summary_lines.append(f"  - {crit} gaps: " + "; ".join(gaps[:2]))

        red_flags = consolidated["overall"].get("global_red_flags") or []
        if red_flags:
            summary_lines.append("- Global red flags: " + "; ".join(red_flags[:3]))
        if top_risks:
            summary_lines.append("- Top risks: " + "; ".join(top_risks[:3]))
        if priority_followups:
            summary_lines.append("- Priority follow-ups: " + "; ".join(priority_followups[:3]))

        reply = (
            "\n".join(summary_lines)
            + "\n\n```json\n"
            + json.dumps(consolidated, indent=2, ensure_ascii=False)
            + "\n```"
        )
        turn_messages.append(ChatMessage(role="assistant", content=reply))
        yield turn_messages

    except Exception as e:
        turn_messages.append(ChatMessage(role="assistant", content=f"⚠️ {type(e).__name__}: {e}"))
        yield turn_messages


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    load_dotenv(verbose=True)
    logging.basicConfig(level=logging.INFO)

    applications_df = load_applications_json(DATASET_CSV_PATH)
    rubric = load_rubric_json(RUBRIC_CSV_PATH)
    store = LocalDataStore(applications_df=applications_df, rubric=rubric)

    worker0_agent, worker1_agent, worker2_agent = build_workers(DEFAULT_MODEL, client_manager.openai_client)

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            ["List applications"],
            ["Evaluate Company 7"],
            ["Evaluate row 12"],
            ["Evaluate 12"],
        ],
        title="Vector One: Agent Experiment 2 (Extractor + Scorer + Auditor)",
        description="Worker0 extracts evidence; Worker1 scores from evidence only; Worker2 audits fairness.",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
