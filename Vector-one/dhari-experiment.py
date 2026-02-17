"""
Vector One: Agent Experiment 1 (Python Orchestrator + Worker 1 Scorer + Worker 2 Evaluator)

Goal:
- Chat with the bot in Gradio
- Ask: "List applications", "Evaluate Company 7", "Evaluate row 12", "Evaluate 12"
- Deterministic (Python) orchestration for data retrieval + selector resolution
- LLM only used for scoring (Worker 1) + verification/adjustment (Worker 2)

Why this version works:
- Avoids LLM tool-call argument JSON issues (e.g., {}{"text":"Company 7"})
- Uses Python to fetch application + rubric reliably, then calls workers with strict JSON prompts

Run:
  python experiment-one.py
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
    # data is a list[dict] (one dict per company)
    return pd.DataFrame(data)

def load_rubric_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _read_csv_robust(path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore


def load_applications(csv_path: str) -> pd.DataFrame:
    df = _read_csv_robust(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_rubric_csv(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    df = _read_csv_robust(csv_path)
    df = df.dropna(how="all")
    df = df.where(pd.notnull(df), None)
    # Keep the same shape your worker instructions expect: rubric_sheets
    return {"rubric": df.to_dict(orient="records")}


def infer_company_identifier_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    """
    Best-effort inference of a company identifier column.
    Skips "Unnamed: ..." columns because those are often index artifacts.
    """
    candidates: List[str] = []
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("unnamed"):
            continue
        if any(k in cl for k in ["company", "organization", "org", "applicant", "name"]):
            candidates.append(c)
    primary = candidates[0] if candidates else None
    return primary, candidates


def application_packet_from_row(df: pd.DataFrame, row_idx: int) -> Dict[str, Any]:
    row = df.iloc[row_idx].to_dict()

    cleaned = {}
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
    """
    Accepts:
      - "row 12"
      - "company 7"  (treated as row 7 in anonymized sets)
      - "12"         (treated as row 12)
      - exact/partial match in inferred company-id columns
    Returns 0-based row index.
    """
    s = selector.strip()

    # row N
    m = re.search(r"\brow\s+(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n < 1 or n > len(df):
            raise ValueError(f"Row {n} out of range (1..{len(df)})")
        return n - 1

    # company N -> row N
    m = re.search(r"\bcompany\s+(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n < 1 or n > len(df):
            raise ValueError(f"Company {n} out of range (1..{len(df)})")
        return n - 1

    # bare integer -> row N
    if re.fullmatch(r"\d+", s):
        n = int(s)
        if n < 1 or n > len(df):
            raise ValueError(f"Row {n} out of range (1..{len(df)})")
        return n - 1

    # match against candidate company cols
    primary_id_col, candidate_cols = infer_company_identifier_columns(df)
    cols_to_try: List[str] = []
    if primary_id_col:
        cols_to_try.append(primary_id_col)
    cols_to_try += [c for c in candidate_cols if c not in cols_to_try]

    for col in cols_to_try:
        series = df[col].astype(str)

        # exact match
        matches = series[series.str.strip() == s]
        if len(matches) == 1:
            return int(matches.index[0])

        # contains match
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


# def _compute_average(final_scores: Dict[str, Any]) -> float:
#     vals: List[float] = []
#     for _, v in final_scores.items():
#         if isinstance(v, dict) and "score" in v:
#             try:
#                 vals.append(float(v["score"]))
#             except Exception:
#                 pass
#     return round(sum(vals) / len(vals), 2) if vals else 0.0

def _compute_average_level(final_scores: Dict[str, Any]) -> float:
    vals: List[float] = []
    for v in final_scores.values():
        if isinstance(v, dict) and "level" in v:
            try:
                vals.append(float(v["level"]))
            except Exception:
                pass
    return round(sum(vals) / len(vals), 2) if vals else 0.0

def _derive_overall_level_from_percent(overall_percent: float, rubric_json: Dict[str, Any]) -> int:
    bands = _get_level_bands(rubric_json)  # {level: (min,max)}

    # choose the level whose band contains the percent
    for level in sorted(bands.keys()):
        lo, hi = bands[level]
        if lo <= overall_percent <= hi:
            return int(level)

    # fallback (shouldn't happen if bands cover 0..100)
    return 0 if overall_percent < 0 else 4


def _compute_average_percent(final_scores: Dict[str, Any]) -> float:
    vals: List[float] = []
    for v in final_scores.values():
        if isinstance(v, dict) and "percent" in v:
            try:
                vals.append(float(v["percent"]))
            except Exception:
                pass
    return round(sum(vals) / len(vals), 1) if vals else 0.0

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

        # enforce band
        if percent < lo or percent > hi:
            old = percent
            percent = max(lo, min(hi, percent))
            fixes.append({"criterion": crit, "issue": "percent_outside_band", "from": old, "to": percent, "band": [lo, hi]})

        obj["level"] = level
        obj["percent"] = percent

    return fixes

def _apply_gap_based_calibration(
    final_scores: Dict[str, Any],
    rubric_json: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    If gaps exist, push percent away from the top of the band.
    - 1 material gap => cap at mid-band
    - 2+ material gaps => cap at bottom-third
    """
    bands = _get_level_bands(rubric_json)
    adjustments: List[Dict[str, Any]] = []

    for crit, obj in final_scores.items():
        if not isinstance(obj, dict):
            continue

        level = int(obj.get("level", 0))
        lo, hi = bands.get(level, (0, 100))
        mid = int(round((lo + hi) / 2))
        bottom_third = int(round(lo + (hi - lo) / 3))

        gaps = obj.get("gaps") or []
        gap_count = len(gaps) if isinstance(gaps, list) else 0

        # cap logic
        old = int(obj.get("percent", mid))
        new = old

        if gap_count >= 2:
            new = min(old, bottom_third)
        elif gap_count == 1:
            new = min(old, mid)

        if new != old:
            obj["percent"] = new
            adjustments.append({
                "criterion": crit,
                "issue": "gap_based_cap",
                "from": old,
                "to": new,
                "band": [lo, hi],
                "gap_count": gap_count
            })

    return adjustments

def _derive_confidence_from_gaps(final_scores: Dict[str, Any]) -> str:
    # count total gaps across criteria
    total_gaps = 0
    criteria_with_gaps = 0
    for v in final_scores.values():
        if isinstance(v, dict):
            gaps = v.get("gaps") or []
            if isinstance(gaps, list) and len(gaps) > 0:
                criteria_with_gaps += 1
                total_gaps += len(gaps)

    # simple rule-of-thumb
    if criteria_with_gaps == 0:
        return "high"
    if criteria_with_gaps <= 2 and total_gaps <= 3:
        return "medium"
    return "low"

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

        # light cleanup (optional)
        chunk = re.sub(r"\n{3,}", "\n\n", chunk).strip()

        # keep chunks reasonably sized
        if chunk:
            sections[key] = chunk[:6000]

    return sections




# ----------------------------
# Local store (Python tools, not exposed to LLM)
# ----------------------------

@dataclass
class LocalDataStore:
    applications_df: pd.DataFrame
    # rubric: Dict[str, List[Dict[str, Any]]]
    rubric: Dict[str, Any]

    def list_applications(self, limit: int = 20) -> Dict[str, Any]:
        primary_id_col, _ = infer_company_identifier_columns(self.applications_df)
        items = []
        for i in range(min(limit, len(self.applications_df))):
            if primary_id_col:
                cid = str(self.applications_df.iloc[i][primary_id_col])
            else:
                cid = f"row_{i+1}"
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

WORKER_1_INSTRUCTIONS = """
You are Worker 1 (Scorer).
You will receive:
- an application packet (structured fields)
- a rubric JSON (rubric_sheets)

Your job:
1) Evaluate the application on EXACTLY these 4 criteria (defined in the rubric):
   - use_case_clarity_alignment
   - dataset_accessibility_suitability
   - infrastructure_tooling_readiness
   - execution_feasibility

2) For each criterion provide BOTH:
   - level: integer from 0 to 4
   - percent: integer from 0 to 100 that MUST fall inside the official score range for that level (use rubric_sheets.levels[level].score_range.min/max)

How to choose level + percent:
- Review the rubric carefully. Each criterion has a detailed description and 5 levels (0-4) with associated score ranges.
- Choose the BEST-matching level (0–4) by comparing the application to the rubric text in: rubric_sheets.rubric[*].readiness_by_level[level]
- Then choose a percent within that level’s band:
    * near the bottom of the band if it barely meets the level
    * mid-band if it clearly meets the level
    * near the top if it strongly matches with minimal gaps
- If evidence is weak or there are important unknowns, LOWER the percent within the band even if the level is correct.  

For each criterion also provide:
- rationale: short, direct justification
- percent_rationale: 1 sentence explaining why the percent is low/mid/high within the level band
- evidence_fields: list of application field keys you used (from application_packet.application)
- gaps: missing info / follow-up questions

Rules:
- Do NOT invent details.
- Base evidence ONLY on the provided application packet fields.
- Reference the rubric best-effort (e.g., mention rubric row/column names if present).
- Output MUST be valid JSON and MUST match the schema below.
- If there is any key unknown dependency mentioned in gaps (e.g., missing dataset source/format, unclear MLOps), keep percent in the lower or mid part of the level band—not the top.
- Percent calibration caps (within a level band):
  * If there is 1 material unknown dependency for this criterion, cap percent at the midpoint of the band.
  * If there are 2+ material unknowns, cap percent at the bottom-third of the band.
  * Only use the top-third of the band if gaps is empty or contains only minor refinements.
- evidence_fields must list the MOST SPECIFIC field keys you used (e.g., dataset_description, infrastructure, team).
- Do NOT use "qa_consolidated" in evidence_fields unless none of the derived fields contain the needed evidence.


Return JSON exactly with this shape:
{
  "company_id": "...",
  "scores": {
    "use_case_clarity_alignment": {
      "level": 0,
      "percent": 0,
      "rationale": "...",
      "percent_rationale": "...",
      "evidence_fields": ["project_summary", "objectives_milestones", "qa_consolidated"],
      "evidence_quotes": ["...10–25 words...", "..."],
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
You are Worker 2 (Evaluator/Verifier).
You will receive:
- application packet
- rubric JSON
- Worker 1 output JSON

Your job:
- Verify Worker 1's level + percent are reasonable against the rubric and supported by evidence_fields.
- Adjust level/percent if rationale/evidence does not justify them.
- Identify missing evidence or rubric mismatches.
- Output MUST be valid JSON and MUST match the schema below.

Rules:
- Do NOT invent details.
- If you adjust a level or percent, explain why and what evidence/rubric mismatch caused the change.
- Prefer conservative scoring when evidence is weak.
- Percent MUST be an integer 0-100 and MUST fall inside the official band for the chosen level: use rubric_sheets.levels[level].score_range.min/max.
- Choose the BEST-matching level by comparing the application to: rubric_sheets.rubric[*].readiness_by_level[level].
- Percent placement within the level band:
  * near the bottom if it barely meets the level
  * mid-band if it clearly meets the level
  * near the top if it strongly matches with minimal gaps
- If a percent is exactly on a boundary shared by two levels (e.g., 25, 50, 70, 90), only choose the higher level if the rubric text clearly matches; otherwise choose the lower level.
- If there is any key unknown dependency mentioned in gaps (e.g., missing dataset source/format, unclear MLOps), keep percent in the lower or mid part of the level band—not the top.
- Worker 2 must add value: if there are material gaps/unknown dependencies, you should EITHER:
  (a) lower the percent within the same level, OR
  (b) lower confidence (high -> medium),
  even if the level stays the same.
- Confidence rubric:
  * high: no material unknowns; evidence is specific and consistent
  * medium: 1–2 material unknowns that could affect delivery
  * low: major unknowns in data access, scope, or feasibility
- Percent calibration caps (within a level band):
  * If there is 1 material unknown dependency for this criterion, cap percent at the midpoint of the band.
  * If there are 2+ material unknowns, cap percent at the bottom-third of the band.
  * Only use the top-third of the band if gaps is empty or contains only minor refinements.
- evidence_fields must list the MOST SPECIFIC field keys you used (e.g., dataset_description, infrastructure, team).
- Do NOT use "qa_consolidated" in evidence_fields unless none of the derived fields contain the needed evidence.


Return JSON exactly with this shape:
{
  "company_id": "...",
  "final_scores": {
    "use_case_clarity_alignment": {
      "level": 0,
      "percent": 0,
      "decision": "accept | adjust",
      "reason": "...",
      "evidence_fields": ["qa_consolidated"],
      "evidence_quotes": ["...10–25 words..."],
      "gaps": ["..."]
    },
    "dataset_accessibility_suitability": { ... },
    "infrastructure_tooling_readiness": { ... },
    "execution_feasibility": { ... }
  },
  "adjustments": [
    {"criterion": "use_case_clarity_alignment", "from_level": 3, "to_level": 2, "from_percent": 78, "to_percent": 66, "reason": "..."}
  ],
  "confidence": "low | medium | high",
  "notes": "short overall notes"
}
"""


def build_workers(model_name: str, openai_client):
    worker1 = agents.Agent(
        name="Worker1_Scorer",
        instructions=WORKER_1_INSTRUCTIONS,
        tools=[],
        model=agents.OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client),
    )

    worker2 = agents.Agent(
        name="Worker2_Evaluator",
        instructions=WORKER_2_INSTRUCTIONS,
        tools=[],
        model=agents.OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client),
    )

    return worker1, worker2


# Globals set in __main__
store: LocalDataStore
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
        selector = q
        selector = re.sub(r"^\s*(evaluate|score)\s+", "", selector, flags=re.IGNORECASE).strip()

        packet = store.get_application(selector)
        rubric_obj = store.get_rubric()

        # Worker 1
        w1_payload = {
            "application_packet": packet,
            "rubric": rubric_obj,
        }
        w1_result = await agents.Runner.run(
            worker1_agent,
            input=json.dumps(w1_payload, ensure_ascii=False),
            session=session,
            max_turns=10,
        )
        w1_json = _safe_json_loads(str(w1_result.final_output))

        # Worker 2
        w2_payload = {
            "application_packet": packet,
            "rubric": rubric_obj,
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
        gap_caps = _apply_gap_based_calibration(final_scores, store.rubric)

        # avg_level = _compute_average_level(final_scores)
        # avg_percent = _compute_average_percent(final_scores)
        avg_percent = _compute_average_percent(final_scores)
        avg_level = _derive_overall_level_from_percent(avg_percent, store.rubric)

        consolidated = {
            "company_id": packet["company_id"],
            "scores": final_scores,
            "overall": {
                "average_level": avg_level,
                "overall_percent": avg_percent,
                # overall rec comes from worker1 (because worker2 schema doesn't include it)
                "recommendation": w1_json.get("overall", {}).get("recommendation", "Proceed with conditions"),
                "top_risks": w1_json.get("overall", {}).get("top_risks", []),
                "priority_followups": w1_json.get("overall", {}).get("priority_followups", []),
                "confidence": w2_json.get("confidence") or _derive_confidence_from_gaps(final_scores),
            },
            "audit": {
                "worker1_output": w1_json,
                "worker2_output": w2_json,
                "adjustments": w2_json.get("adjustments", []),
                "percent_fixes": percent_fixes,
                "gap_caps": gap_caps,
            },
        }

        summary_lines = [
            f"Evaluation for **{packet['company_id']}** (row {packet['row_index_1_based']}):",
            # f"- Average score: **{avg}/4**",
            f"- Average level: **{avg_level}/4**",
            f"- Overall percent: **{avg_percent}%**",
            f"- Recommendation: **{consolidated['overall']['recommendation']}**",
            f"- Confidence: **{consolidated['overall']['confidence']}**",
        ]
        # Add per-criterion breakdown
        for crit, v in final_scores.items():
            if isinstance(v, dict):
                summary_lines.append(
                    f"- {crit}: Level **{v.get('level')}**, **{v.get('percent')}%**"
                )
                # show top gaps if they exist
                gaps = v.get("gaps") or [] 
                if gaps:
                    summary_lines.append(
                        f"  - {crit} gaps: " + "; ".join(gaps[:2])
                    )
        if consolidated["overall"]["top_risks"]:
            summary_lines.append("- Top risks: " + "; ".join(consolidated["overall"]["top_risks"][:3]))
        if consolidated["overall"]["priority_followups"]:
            summary_lines.append("- Priority follow-ups: " + "; ".join(consolidated["overall"]["priority_followups"][:3]))

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

    # Load local data
    applications_df = load_applications_json(DATASET_CSV_PATH)
    rubric = load_rubric_json(RUBRIC_CSV_PATH)
    store = LocalDataStore(applications_df=applications_df, rubric=rubric)

    # Build worker agents
    worker1_agent, worker2_agent = build_workers(DEFAULT_MODEL, client_manager.openai_client)

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            ["List applications"],
            ["Evaluate Company 7"],
            ["Evaluate row 12"],
            ["Evaluate 12"],
        ],
        title="Vector One: Agent Experiment 1 (Python Orchestrator + Scorer + Evaluator)",
        description="Local-only scoring using CSV applications + CSV rubric. No web search / no KB.",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())