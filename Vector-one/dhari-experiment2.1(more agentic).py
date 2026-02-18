"""
Vector One: Agent Experiment 2.1 (More Agentic + More Reliable)

Implements:
- Deterministic-ish runs (temperature=0 + optional seed + stable prompts)
- Percent quantization to 3 anchors per level band (low/mid/high) to reduce score drift
- Worker2 must choose judgement: too_high/fair/too_low and MUST adjust if not fair
- Bounded agentic loop: Worker1 <-> Worker2 can iterate (max 2 rounds) until all fair
- Orchestrator computes overall percent/level deterministically + rule-based final recommendation
- Optional trigger: improvement plan generator when overall is weak or confidence low (Worker3)

Run:
  python experiment-two.py
"""

import asyncio
import json
import logging
import os
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

# Reliability knobs (reduce drift)
LLM_TEMPERATURE = float(os.getenv("VECTOR_ONE_TEMPERATURE", "0"))
LLM_TOP_P = float(os.getenv("VECTOR_ONE_TOP_P", "1"))
LLM_SEED = os.getenv("VECTOR_ONE_SEED")  # optional; string -> int if present
MAX_AUDIT_ROUNDS = int(os.getenv("VECTOR_ONE_MAX_AUDIT_ROUNDS", "2"))

# Percent quantization anchors (per level band)
ANCHOR_OFFSET = int(os.getenv("VECTOR_ONE_ANCHOR_OFFSET", "5"))  # within band


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

def _percent_anchors_for_level(level: int, rubric_json: Dict[str, Any]) -> Tuple[int, int, int]:
    lo, hi = _get_level_bands(rubric_json).get(level, (0, 100))
    mid = int(round((lo + hi) / 2))
    low_anchor = min(hi, lo + ANCHOR_OFFSET)
    high_anchor = max(lo, hi - ANCHOR_OFFSET)
    return low_anchor, mid, high_anchor

def _quantize_percent(percent: int, level: int, rubric_json: Dict[str, Any]) -> int:
    low_anchor, mid, high_anchor = _percent_anchors_for_level(level, rubric_json)
    choices = [low_anchor, mid, high_anchor]
    return min(choices, key=lambda x: abs(x - percent))

def _validate_and_quantize_level_percent(scores: Dict[str, Any], rubric_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    - clamp level to 0..4
    - clamp percent to 0..100
    - enforce band for that level
    - quantize percent to 3 anchors (low/mid/high) for stability
    """
    bands = _get_level_bands(rubric_json)
    fixes: List[Dict[str, Any]] = []

    for crit, obj in scores.items():
        if not isinstance(obj, dict):
            continue

        level = _clamp_int(obj.get("level"), 0, 4, default=0)
        lo, hi = bands.get(level, (0, 100))
        mid = int(round((lo + hi) / 2))

        percent_raw = obj.get("percent")
        percent = _clamp_int(percent_raw, 0, 100, default=mid)

        # Enforce band
        if percent < lo or percent > hi:
            old = percent
            percent = max(lo, min(hi, percent))
            fixes.append({"criterion": crit, "issue": "percent_outside_band", "from": old, "to": percent, "band": [lo, hi]})

        # Quantize to anchors
        q_old = percent
        percent = _quantize_percent(percent, level, rubric_json)
        if percent != q_old:
            fixes.append({
                "criterion": crit,
                "issue": "percent_quantized",
                "from": q_old,
                "to": percent,
                "anchors": list(_percent_anchors_for_level(level, rubric_json)),
                "band": [lo, hi],
            })

        obj["level"] = level
        obj["percent"] = percent

    return fixes

def _compute_average_percent(scores: Dict[str, Any]) -> float:
    vals: List[float] = []
    for v in scores.values():
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

def _downshift_recommendation(rec: str, steps: int = 1) -> str:
    order = ["Proceed", "Proceed with conditions", "Hold", "Reject"]
    try:
        i = order.index(rec)
    except ValueError:
        i = 1  # default to "Proceed with conditions"
    return order[min(len(order) - 1, i + max(0, steps))]

def _rule_based_recommendation(overall_level: int, confidence: str, red_flags: List[str]) -> str:
    # base recommendation from overall level
    if overall_level >= 4:
        rec = "Proceed"
    elif overall_level == 3:
        rec = "Proceed with conditions"
    elif overall_level == 2:
        rec = "Hold"
    else:
        rec = "Reject"

    # downshift on low confidence
    if confidence == "low":
        rec = _downshift_recommendation(rec, 1)

    # downshift if global red flags exist
    if red_flags:
        rec = _downshift_recommendation(rec, 1)

    return rec


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
- Do NOT score. Do NOT assign levels or percents. Do NOT say "strong/mature/weak" as a rating.
- Do NOT invent details.
- Use the MOST SPECIFIC derived fields when possible (e.g., dataset_description, infrastructure, team).
- Only use qa_consolidated if the needed evidence is not available in derived fields.
- Provide concise quotes (10–25 words) that clearly support evidence.
- Identify gaps / unknown dependencies per criterion. If a gap is "material", label it explicitly as "MATERIAL:".

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

You MUST output:
- level: integer 0–4
- percent: integer 0–100 AND MUST be inside the official band for that level:
  rubric_sheets.levels[level].score_range.min/max

Stability rule (IMPORTANT):
For a given level, percent MUST be one of THREE anchors:
- low_anchor (near bottom of band)
- mid_anchor (mid band)
- high_anchor (near top of band)
The orchestrator will enforce anchors, but you should choose anchors directly.

How to score:
- Choose BEST-matching level by comparing evidence_pack to:
  rubric_sheets.rubric[*].readiness_by_level[level]
- Choose percent anchor:
  * low_anchor if barely meets level OR any MATERIAL gap exists
  * mid_anchor if clearly meets but has minor gaps
  * high_anchor only if gaps is empty or purely minor refinements

Hard rule:
- If gaps contains any MATERIAL unknown dependency for that criterion, do NOT use high_anchor.

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

Hard requirements:
1) For each criterion choose judgement: "too_high | fair | too_low"
2) If judgement is "too_high" or "too_low", you MUST adjust level and/or percent.
   - decision must be "adjust"
   - If judgement is "fair", decision must be "accept"
3) Percent MUST be within level band and SHOULD use an anchor (low/mid/high).
4) If Worker0 gaps include "MATERIAL:" for a criterion, the score cannot be top-of-band.
5) Your reasons must explicitly cite the gaps and/or missing evidence.

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

WORKER_3_INSTRUCTIONS = """
You are Worker 3 (Improvement Plan Generator).

You will receive:
- rubric JSON (rubric_sheets)
- evidence_pack (from Worker0)
- final_scores (from Worker2)
- overall: overall_level, overall_percent, confidence, global_red_flags

Goal:
Create a short, concrete improvement plan to move this application up at least one overall level.

Rules:
- Do NOT rescore.
- Provide checklist items mapped to the four criteria.
- Each checklist item must be phrased as an action + what artifact/evidence to provide.
- Keep it short and practical.

Return JSON:
{
  "company_id": "...",
  "improvement_plan": {
    "use_case_clarity_alignment": ["..."],
    "dataset_accessibility_suitability": ["..."],
    "infrastructure_tooling_readiness": ["..."],
    "execution_feasibility": ["..."],
    "global": ["..."]
  }
}
"""

def _maybe_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def build_model(model_name: str, openai_client):
    kwargs = {
        "model": model_name,
        "openai_client": openai_client,
    }
    # Best-effort: only set if supported by your agents wrapper.
    # If unsupported, it will raise TypeError; we handle that in build_workers.
    kwargs["temperature"] = LLM_TEMPERATURE
    kwargs["top_p"] = LLM_TOP_P
    seed_val = _maybe_int(LLM_SEED)
    if seed_val is not None:
        kwargs["seed"] = seed_val
    return agents.OpenAIChatCompletionsModel(**kwargs)

def build_workers(model_name: str, openai_client):
    # If your OpenAIChatCompletionsModel wrapper doesn't accept temperature/top_p/seed,
    # fall back to minimal constructor.
    try:
        model = build_model(model_name, openai_client)
    except TypeError:
        model = agents.OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client)

    worker0 = agents.Agent(
        name="Worker0_Extractor",
        instructions=WORKER_0_INSTRUCTIONS,
        tools=[],
        model=model,
    )
    worker1 = agents.Agent(
        name="Worker1_Scorer",
        instructions=WORKER_1_INSTRUCTIONS,
        tools=[],
        model=model,
    )
    worker2 = agents.Agent(
        name="Worker2_Auditor",
        instructions=WORKER_2_INSTRUCTIONS,
        tools=[],
        model=model,
    )
    worker3 = agents.Agent(
        name="Worker3_ImprovementPlan",
        instructions=WORKER_3_INSTRUCTIONS,
        tools=[],
        model=model,
    )
    return worker0, worker1, worker2, worker3


# Globals set in __main__
store: LocalDataStore
worker0_agent: agents.Agent
worker1_agent: agents.Agent
worker2_agent: agents.Agent
worker3_agent: agents.Agent


def _needs_improvement_plan(overall_level: int, confidence: str, red_flags: List[str]) -> bool:
    if overall_level <= 2:
        return True
    if confidence == "low":
        return True
    if red_flags:
        return True
    return False

def _all_fair(final_scores: Dict[str, Any]) -> bool:
    for v in final_scores.values():
        if isinstance(v, dict):
            if v.get("judgement") != "fair":
                return False
    return True


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
        # Agentic loop: Worker1 score <-> Worker2 audit (bounded rounds)
        # -------------------------
        w1_json: Dict[str, Any] = {}
        w2_json: Dict[str, Any] = {}
        loop_audit_trail: List[Dict[str, Any]] = []

        worker1_feedback: Optional[str] = None

        for round_idx in range(MAX_AUDIT_ROUNDS):
            # Worker 1: score evidence_pack (+ optional targeted feedback)
            w1_payload = {
                "company_id": packet["company_id"],
                "rubric": rubric_obj,
                "evidence_pack": evidence_pack,
            }
            if worker1_feedback:
                w1_payload["audit_feedback"] = worker1_feedback

            w1_result = await agents.Runner.run(
                worker1_agent,
                input=json.dumps(w1_payload, ensure_ascii=False),
                session=session,
                max_turns=10,
            )
            w1_json = _safe_json_loads(str(w1_result.final_output))

            # Worker 2: audit Worker 1
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

            # Orchestrator enforces validity + quantization for stability
            fixes = _validate_and_quantize_level_percent(final_scores, store.rubric)

            loop_audit_trail.append({
                "round": round_idx + 1,
                "worker1_output": w1_json,
                "worker2_output": w2_json,
                "orchestrator_fixes": fixes,
            })

            # stop if all fair
            if _all_fair(final_scores):
                break

            # targeted feedback back to Worker1: only criteria not fair
            not_fair = []
            for crit, obj in final_scores.items():
                if isinstance(obj, dict) and obj.get("judgement") != "fair":
                    not_fair.append(crit)

            worker1_feedback = (
                "Re-score ONLY the criteria listed as not fair by the auditor. "
                "Use the rubric + evidence_pack + MATERIAL gaps rules. "
                f"Criteria not fair: {', '.join(not_fair)}. "
                "If any criterion has a MATERIAL gap, do not choose top-of-band."
            )

        # Use final round's audited scores
        final_scores = w2_json.get("final_scores", {})
        fixes_final = _validate_and_quantize_level_percent(final_scores, store.rubric)

        overall_percent = _compute_average_percent(final_scores)
        overall_level = _derive_overall_level_from_percent(overall_percent, store.rubric)

        global_red_flags = w0_json.get("global_red_flags", []) or []
        confidence = (w2_json.get("confidence") or _derive_confidence_from_gaps(final_scores)).lower()
        confidence = confidence if confidence in ["low", "medium", "high"] else "medium"

        # Worker1 provides rec, orchestrator finalizes deterministically
        worker1_rec = w1_json.get("overall", {}).get("recommendation", "Proceed with conditions")
        final_rec = _rule_based_recommendation(overall_level, confidence, global_red_flags)

        # keep risks/followups from Worker1 (scorer), but you can also merge with Worker0 gaps if desired
        top_risks = w1_json.get("overall", {}).get("top_risks", [])
        priority_followups = w1_json.get("overall", {}).get("priority_followups", [])

        consolidated: Dict[str, Any] = {
            "company_id": packet["company_id"],
            "scores": final_scores,
            "overall": {
                "overall_level": overall_level,
                "overall_percent": overall_percent,
                "recommendation": final_rec,
                "worker1_recommendation": worker1_rec,
                "confidence": confidence,
                "top_risks": top_risks,
                "priority_followups": priority_followups,
                "global_red_flags": global_red_flags,
            },
            "audit": {
                "worker0_evidence": w0_json,
                "loop_rounds": loop_audit_trail,
                "final_worker1_scoring": w1_json,
                "final_worker2_audit": w2_json,
                "final_orchestrator_fixes": fixes_final,
                "settings": {
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P,
                    "seed": _maybe_int(LLM_SEED),
                    "max_audit_rounds": MAX_AUDIT_ROUNDS,
                    "anchor_offset": ANCHOR_OFFSET,
                },
            },
        }

        # Optional: improvement plan trigger
        if _needs_improvement_plan(overall_level, confidence, global_red_flags):
            w3_payload = {
                "company_id": packet["company_id"],
                "rubric": rubric_obj,
                "evidence_pack": evidence_pack,
                "final_scores": final_scores,
                "overall": consolidated["overall"],
            }
            w3_result = await agents.Runner.run(
                worker3_agent,
                input=json.dumps(w3_payload, ensure_ascii=False),
                session=session,
                max_turns=10,
            )
            w3_json = _safe_json_loads(str(w3_result.final_output))
            consolidated["improvement_plan"] = w3_json.get("improvement_plan", {})

        # -------------------------
        # Render
        # -------------------------
        summary_lines = [
            f"Evaluation for **{packet['company_id']}** (row {packet['row_index_1_based']}):",
            f"- Overall level: **{overall_level}/4**",
            f"- Overall percent: **{overall_percent}%**",
            f"- Recommendation: **{final_rec}** (Worker1 suggested: {worker1_rec})",
            f"- Confidence: **{confidence}**",
        ]

        for crit, v in final_scores.items():
            if isinstance(v, dict):
                summary_lines.append(
                    f"- {crit}: Level **{v.get('level')}**, **{v.get('percent')}%** ({v.get('judgement','')})"
                )
                gaps = v.get("gaps") or []
                if gaps:
                    summary_lines.append(f"  - {crit} gaps: " + "; ".join(gaps[:2]))

        if global_red_flags:
            summary_lines.append("- Global red flags: " + "; ".join(global_red_flags[:3]))
        if top_risks:
            summary_lines.append("- Top risks: " + "; ".join(top_risks[:3]))
        if priority_followups:
            summary_lines.append("- Priority follow-ups: " + "; ".join(priority_followups[:3]))

        if "improvement_plan" in consolidated and consolidated["improvement_plan"]:
            summary_lines.append("- Improvement plan generated (see JSON).")

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

    worker0_agent, worker1_agent, worker2_agent, worker3_agent = build_workers(
        DEFAULT_MODEL, client_manager.openai_client
    )

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            ["List applications"],
            ["Evaluate Company 7"],
            ["Evaluate row 12"],
            ["Evaluate 12"],
        ],
        title="Vector One: Agent Experiment 2.1 (Extractor + Scorer + Auditor + Loop + Quantized Scores)",
        description="Worker0 extracts evidence; Worker1 scores from evidence only; Worker2 audits; orchestrator loops if not fair and stabilizes scoring.",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
