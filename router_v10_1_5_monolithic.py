"""
# Architecte & Author : M.B.E.F from Benslaiman.com Contact Email: contact@benslaiman.com  
# AI-assisted desing & Implementation (used as cognitive accelerator)
# Date Starting of this project  08/01/2026
# Date Creation of this Script  27/02/2026

LAB LLM ROUTER v10.1.5 Telecom UIHelperIsolated — Structured Memory (importance + decay) + Conditional Time + Request Logging
Based on v7.5.1 behaviors:
- SCCI-Router + tagged models: "[node] model" in /v1/models
- bucketed routing + score-based selection + sticky models
- router-mode anti-thrash penalties
- discovery loop + optional /ready + /metrics probing
- Conditional time injection (ONLY when needed, and NEVER for OpenWebUI meta-tasks)
- Safe request logging (ROUTER_LOG_REQUESTS=0/1/2/3)
- Memory poison protection (don’t store OpenWebUI meta prompts / huge blobs)

v8 Memory changes:
- New table: memories_v8 (structured)
- Fields: kind, importance, access_count, last_accessed, confidence
- Retrieval uses decay score:
    score = decayed_importance(age) + reinforcement(access_count) + confidence_term
- When memories are returned, they are reinforced:
    access_count++, last_accessed=now, importance+=RECALL_BOOST (capped)
- Auto-migration from old table "memories" -> "memories_v8" if needed.
- No need to delete DB.

Run:
  set ROUTER_LOG_REQUESTS=1
  set 

ROUTER_LOG_MEMORY_WRITES=1
  uvicorn router_app_Brain_v8_with_structured_memory_decay:app --host 0.0.0.0 --port 8000
"""



# ================================
# SCCI v10.1.5 Telecom UIHelperIsolated
# ================================
# Deterministic medical vision routing target
MEDGEMMA_DEFAULT_NODE = "B-GPU0"
MEDGEMMA_DEFAULT_MODEL = "medgemma1_5-4b-vision@2k-q4km"

# Demo medical disclaimer (safe single‑line string to avoid syntax errors)
SCCI_MEDICAL_DEMO_DISCLAIMER_IN = (
    "Demo notice: This system can describe visible elements in medical images "
    "for research or educational demonstration only. It is not a medical "
    "diagnostic system and does not provide medical advice."
)


# Optional: log selected incoming HTTP headers (useful for OpenWebUI forwarded IDs)
import os
from typing import Any, Dict, List, Optional, Tuple

FUNCTION_TOOL_NODE_PRIMARY = os.getenv("FUNCTION_TOOL_NODE_PRIMARY", "B-PHONE2").strip()
FUNCTION_TOOL_NODE_FALLBACK = os.getenv("FUNCTION_TOOL_NODE_FALLBACK", "B-CPU0").strip()
FUNCTION_TOOL_MODEL_ID = os.getenv("FUNCTION_TOOL_MODEL_ID", "qwen2.5-0.5b-instruct@1k-q4km").strip()
UI_HELPER_NODE_PRIMARY = os.getenv("UI_HELPER_NODE_PRIMARY", "B-PHONE1").strip()
UI_HELPER_NODE_FALLBACK = os.getenv("UI_HELPER_NODE_FALLBACK", "B-CPU0").strip()
UI_HELPER_MODEL_ID = os.getenv("UI_HELPER_MODEL_ID", "qwen2.5-coder-1.5b-instruct-q5_km").strip()
UI_HELPER_TRIM_MAX_CHARS = int(os.getenv("UI_HELPER_TRIM_MAX_CHARS", "600"))
UI_HELPER_CHAT_HISTORY_MAX_CHARS = int(os.getenv("UI_HELPER_CHAT_HISTORY_MAX_CHARS", "120"))
TIME_TOOL_ENABLE = int(os.getenv("TIME_TOOL_ENABLE", "1"))
TIME_TOOL_CITIES = [c.strip() for c in os.getenv("TIME_TOOL_CITIES", "Paris,Tokyo,New York").split(",") if c.strip()]
CHAT_LADDER_MODELS = [m.strip() for m in os.getenv("CHAT_LADDER_MODELS", "gemma3-1b-chat@2k-q8_0,gemma3-1b-chat@4k-q8_0,gemma3-1b-chat@8k-q8_0,gemma3-1b-chat@16k-q8_0,gemma3-1b-chat@32k-q8_0").split(",") if m.strip()]
CHAT_LADDER_NODES = [n.strip() for n in os.getenv("CHAT_LADDER_NODES", "B-PHONE0,B-PHONE3,B-CPU0,B-GPU0").split(",") if n.strip()]
LONGTEXT_DIRECT_THRESHOLD = int(os.getenv("LONGTEXT_DIRECT_THRESHOLD", "1500"))
LONGTEXT_HISTORY_MESSAGES = int(os.getenv("LONGTEXT_HISTORY_MESSAGES", "4"))
LONGTEXT_HISTORY_CHAR_BUDGET = int(os.getenv("LONGTEXT_HISTORY_CHAR_BUDGET", "12000"))
TOOLS_TRIM_ENABLED = int(os.getenv("TOOLS_TRIM_ENABLED", "1"))
TOOLS_TRIM_MAX_CHARS = int(os.getenv("TOOLS_TRIM_MAX_CHARS", "1600"))
TOOLS_CHAT_HISTORY_MAX_CHARS = int(os.getenv("TOOLS_CHAT_HISTORY_MAX_CHARS", "700"))
LONGTEXT_ESCALATION_BUFFER_RATIO = float(os.getenv("LONGTEXT_ESCALATION_BUFFER_RATIO", "1.25"))
LONGTEXT_EMERGENCY_TOKENS = int(os.getenv("LONGTEXT_EMERGENCY_TOKENS", "12000"))
ROUTER_OWUI_HELPER_MIN_SCORE = int(os.getenv("ROUTER_OWUI_HELPER_MIN_SCORE", "5"))
ROUTER_TOOL_CTX_SAFE_RATIO = float(os.getenv("ROUTER_TOOL_CTX_SAFE_RATIO", "0.60"))
ROUTER_TOOL_LARGE_PROMPT_CHAT_FALLBACK = int(os.getenv("ROUTER_TOOL_LARGE_PROMPT_CHAT_FALLBACK", "1"))
ROUTER_LOG_HEADERS = os.getenv("ROUTER_LOG_HEADERS", "0").strip().lower() in {"1","true","yes"}

ROUTER_LOG_HEADERS_VERBOSE = os.getenv("ROUTER_LOG_HEADERS_VERBOSE", "0").strip().lower() in {"1","true","yes"}
def _debug_log_headers(req: "Request") -> None:
    """
    Debug header logger.

    - Enabled when ROUTER_LOG_HEADERS is truthy (default: on).
    - PII (name/email) is only logged when ROUTER_LOG_HEADERS_VERBOSE is truthy (default: on).
    """
    if not ROUTER_LOG_HEADERS:
        return
    try:
        # Non-PII headers we allow by default
        allowed = {
            "host",
            "user-agent",
            "x-request-id",
            "x-session-id",
            "x-chat-id",
            "x-conversation-id",
            "x-openwebui-chat-id",
            "x-openwebui-message-id",
            "x-openwebui-user-id",
            "x-openwebui-user-role",
        }

        # PII headers (only when verbose)
        sensitive = {
            "x-openwebui-user-name",
            "x-openwebui-user-email",
        }

        out: Dict[str, str] = {}
        for k, v in req.headers.items():
            kl = k.lower()

            # avoid leaking secrets
            if kl in ("authorization", "cookie", "set-cookie", "x-api-key"):
                continue

            if kl in allowed or (ROUTER_LOG_HEADERS_VERBOSE and kl in sensitive):
                out[k] = v

        log.info("[reqhdr] %s", json.dumps(out, ensure_ascii=False))
    except Exception:
        # Never break requests because of debug logging
        pass


import os
import re

# ===== v9.7.7 HYBRID + SEMANTIC =====
def _v977_merge_classification(heuristic, llm):
    if llm == "complex_code":
        return "complex_code"
    if heuristic == "simple_code" and llm == "medium_code":
        return "medium_code"
    return heuristic

def _v977_semantic_boost(text, current):
    t = str(text or "").lower()
    if ("design" in t and ("router" in t or "system" in t)) or "architecture" in t:
        return "complex_code"
    if "distributed" in t:
        return "complex_code"
    return current


# ===== v9.7.6 FAST CLASSIFIER (Qwen 1.5B) =====
CLASSIFIER_MODEL = "qwen2.5-coder-1.5b-instruct"
CLASSIFIER_NODE_PRIMARY = "B-CPU0"
CLASSIFIER_NODE_FALLBACK = "B-PHONE3"
CLASSIFIER_MAX_TOKENS = 32
CLASSIFIER_TEMPERATURE = 0.2
CLASSIFIER_TOP_P = 0.8
CLASSIFIER_TIMEOUT = 3.0

def _build_classifier_prompt(user_text):
    return f"""
Classify the following request into one label:

- simple_code
- medium_code
- complex_code

Respond ONLY with the label.

Request:
{str(user_text)[:600]}
"""

def _normalize_label(text):
    t = str(text).lower()
    if "simple" in t:
        return "simple_code"
    if "complex" in t:
        return "complex_code"
    if "medium" in t:
        return "medium_code"
    return "medium_code"

async def _v976_run_classifier(req_id, user_text):
    prompt = _build_classifier_prompt(user_text)
    body = {
        "model": CLASSIFIER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CLASSIFIER_MAX_TOKENS,
        "temperature": CLASSIFIER_TEMPERATURE,
        "top_p": CLASSIFIER_TOP_P,
        "stream": False,
    }
    try:
        result = await _call_node(CLASSIFIER_NODE_PRIMARY, body, timeout=CLASSIFIER_TIMEOUT)
    except Exception:
        result = await _call_node(CLASSIFIER_NODE_FALLBACK, body, timeout=CLASSIFIER_TIMEOUT)
    out = result["choices"][0]["message"]["content"]
    return _normalize_label(out)


# ===== v9.7.5 UI HELPER ISOLATION =====
def _v975_ui_helper_allowed(user_text, prompt_chars, coding_intent, file_code_intent):
    if coding_intent == 1 or file_code_intent == 1:
        return False, "coding_or_file_code"
    if prompt_chars > 1200:
        return False, "too_long"
    UI_KEYWORDS = [
        "continue","follow","next",
        "rewrite","shorten","summarize",
        "format","fix grammar",
        "title","improve wording"
    ]
    t = str(user_text or "").lower()
    if not any(k in t for k in UI_KEYWORDS):
        return False, "not_ui_task"
    return True, "ok"

def _v975_trim_ui_messages(messages, max_chars=800):
    msgs = messages[-2:]
    text = ""
    out = []
    for m in msgs:
        content = str(m.get("content",""))
        if len(text) + len(content) > max_chars:
            content = content[:max_chars-len(text)]
        text += content
        out.append({"role": m.get("role"), "content": content})
    return out

def _v975_select_ui_node():
    primary = "B-PHONE1"
    fallback = "B-CPU0"
    n = _nodes.get(primary)
    if n:
        try:
            if n.healthy_cached() and not _node_looks_busy(n):
                return primary
        except:
            pass
    return fallback

# ---------------- SICC/SRVCC-inspired model-centric ctx escalation ----------------
# Defaults (work without env). Env overrides are optional.
ROUTER_CTX_PREWARM_RATIO = float(os.getenv("ROUTER_CTX_PREWARM_RATIO", "0.70"))
ROUTER_CTX_HANDOVER_RATIO = float(os.getenv("ROUTER_CTX_HANDOVER_RATIO", "0.82"))
ROUTER_CTX_COMPLETION_RESERVE = int(os.getenv("ROUTER_CTX_COMPLETION_RESERVE", "256"))
ROUTER_PREWARM_ENABLED = int(os.getenv("ROUTER_PREWARM_ENABLED", "1"))
ROUTER_PREWARM_TIMEOUT_MS = int(os.getenv("ROUTER_PREWARM_TIMEOUT_MS", "1500"))
ROUTER_PREWARM_LOG = int(os.getenv("ROUTER_PREWARM_LOG", "1"))
ROUTER_PREWARM_COOLDOWN_S = int(os.getenv("ROUTER_PREWARM_COOLDOWN_S", "30"))
ROUTER_PREWARM_PHONE_RETRY_ATTEMPTS = max(1, int(os.getenv("ROUTER_PREWARM_PHONE_RETRY_ATTEMPTS", "3")))
ROUTER_PREWARM_PHONE_RETRY_DELAYS_S = [float(x.strip()) for x in os.getenv("ROUTER_PREWARM_PHONE_RETRY_DELAYS_S", "1.5,3.0").split(",") if x.strip()]
ROUTER_PREWARM_PHONE_WAKE_NODES = {x.strip().upper() for x in os.getenv("ROUTER_PREWARM_PHONE_WAKE_NODES", "B-PHONE0,B-PHONE1,B-PHONE2,B-PHONE3").split(",") if x.strip()}

# v9 coding lane: task-scoped routing + classifier-assisted complexity selection.
# Policy:
# - simple_code  -> B-PHONE3 primary (1.5B coder)
# - medium_code  -> B-GPU0 primary (3B coder ctx ladder)
# - complex_code -> B-GPU0 primary (DeepSeek reason ctx ladder)
# - B-CPU0 is fallback only when preferred nodes are busy/unavailable.
CODER_SIMPLE_MODEL_4K = os.getenv("ROUTER_CODER_SIMPLE_MODEL_4K", "qwen2.5-coder-1.5b-instruct@4k-q5km").strip()
CODER_MEDIUM_MODEL_4K = os.getenv("ROUTER_CODER_MEDIUM_MODEL_4K", "qwen2.5-coder-3b-instruct@4k-q4km").strip()
CODER_MEDIUM_MODEL_8K = os.getenv("ROUTER_CODER_MEDIUM_MODEL_8K", "qwen2.5-coder-3b-instruct@8k-q4km").strip()
CODER_MEDIUM_MODEL_16K = os.getenv("ROUTER_CODER_MEDIUM_MODEL_16K", "qwen2.5-coder-3b-instruct@16k-q4km").strip()
CODER_MEDIUM_MODEL_32K = os.getenv("ROUTER_CODER_MEDIUM_MODEL_32K", "qwen2.5-coder-3b-instruct@32k-q4km").strip()
CODER_COMPLEX_MODEL_8K = os.getenv("ROUTER_CODER_COMPLEX_MODEL_8K", "deepseekr1-8b-reason@8k-q5km").strip()
CODER_COMPLEX_MODEL_16K = os.getenv("ROUTER_CODER_COMPLEX_MODEL_16K", "deepseekr1-8b-reason@16k-hybrid-q5km").strip()
CODER_COMPLEX_MODEL_32K = os.getenv("ROUTER_CODER_COMPLEX_MODEL_32K", "deepseekr1-8b-reason@32k-hybrid-q5km").strip()
CODER_NODE_SIMPLE_PRIMARY = os.getenv("ROUTER_CODER_NODE_SIMPLE_PRIMARY", "B-PHONE3").strip()
CODER_NODE_SIMPLE_FALLBACK = os.getenv("ROUTER_CODER_NODE_SIMPLE_FALLBACK", "B-CPU0").strip()
CODER_NODE_PRIMARY = os.getenv("ROUTER_CODER_NODE_PRIMARY", "B-GPU0").strip()
CODER_NODE_FALLBACK = os.getenv("ROUTER_CODER_NODE_FALLBACK", "B-CPU0").strip()

QWEN35_4B_MODEL = os.getenv("ROUTER_QWEN35_4B_MODEL", "qwen3.5-4b-chat@8k-q5km").strip()
QWEN35_NODE_PRIMARY = os.getenv("ROUTER_QWEN35_NODE_PRIMARY", "B-GPU0").strip()
QWEN35_NODE_FALLBACK = os.getenv("ROUTER_QWEN35_NODE_FALLBACK", "B-CPU0").strip()
CODING_CLASSIFIER_NODE = os.getenv("ROUTER_CODING_CLASSIFIER_NODE", "B-CPU0").strip()
CODING_CLASSIFIER_MODEL = os.getenv("ROUTER_CODING_CLASSIFIER_MODEL", CODER_SIMPLE_MODEL_4K).strip()
CODING_CLASSIFIER_TIMEOUT_S = float(os.getenv("ROUTER_CODING_CLASSIFIER_TIMEOUT_S", "8.0"))
CODING_CLASSIFIER_MAX_INPUT_CHARS = int(os.getenv("ROUTER_CODING_CLASSIFIER_MAX_INPUT_CHARS", "800"))
CODING_CLASSIFIER_TEMPERATURE = float(os.getenv("ROUTER_CODING_CLASSIFIER_TEMPERATURE", "0"))
CODING_CLASSIFIER_TOP_K = int(os.getenv("ROUTER_CODING_CLASSIFIER_TOP_K", "1"))
CODING_CLASSIFIER_TOP_P = float(os.getenv("ROUTER_CODING_CLASSIFIER_TOP_P", "1.0"))
CODING_CLASSIFIER_REPEAT_PENALTY = float(os.getenv("ROUTER_CODING_CLASSIFIER_REPEAT_PENALTY", "1.0"))
CODING_CLASSIFIER_MAX_TOKENS = int(os.getenv("ROUTER_CODING_CLASSIFIER_MAX_TOKENS", "3"))
CODING_CLASSIFIER_STOP = [x for x in [s.strip() for s in os.getenv("ROUTER_CODING_CLASSIFIER_STOP", "\\n").split("|")] if x]
CODING_TOOL_CHAR_BUDGET = int(os.getenv("ROUTER_CODING_TOOL_CHAR_BUDGET", "12000"))
CODING_SIMPLE_PROMPT_CHAR_MAX = int(os.getenv("ROUTER_CODING_SIMPLE_PROMPT_CHAR_MAX", "900"))
CODING_SIMPLE_CODE_CHAR_MAX = int(os.getenv("ROUTER_CODING_SIMPLE_CODE_CHAR_MAX", "2200"))
CODING_COMPLEX_PROMPT_CHAR_MIN = int(os.getenv("ROUTER_CODING_COMPLEX_PROMPT_CHAR_MIN", "2200"))
CODING_COMPLEX_CODE_CHAR_MIN = int(os.getenv("ROUTER_CODING_COMPLEX_CODE_CHAR_MIN", "7000"))
CODING_GPU_BUSY_QUEUE_DEPTH = int(os.getenv("ROUTER_CODING_GPU_BUSY_QUEUE_DEPTH", "2"))
CODING_GPU_BUSY_ACTIVE_REQUESTS = int(os.getenv("ROUTER_CODING_GPU_BUSY_ACTIVE_REQUESTS", "2"))

def _parse_ctx_from_model_id(model_id: str) -> int:
    """Parse ctx from alias like '...@8k-...' -> 8192. Returns 0 if unknown."""
    if not model_id:
        return 0
    m = re.search(r"@\s*(\d+)\s*k\b", str(model_id), flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1)) * 1024
    except Exception:
        return 0

def _extract_text_for_estimate(msgs):
    parts = []
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p, dict) and "text" in p and isinstance(p.get("text"), str):
                    parts.append(p.get("text"))
    return "\n".join(parts)

def _last_user_text(msgs):
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                out=[]
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        out.append(p["text"])
                return "\n".join(out)
    return ""

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))

_CODE_PAT = re.compile(
    r"```|\b(def|class|import|from|#include|SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|function\s*\(|console\.log|print\(|System\.out\.println)\b",
    re.IGNORECASE
)



def _pick_chat_ladder_model(tokens_needed: int) -> str:
    want = int(tokens_needed or 0)
    cands = CHAT_LADDER_MODELS or [DEEP_CPU_MODEL_ID]
    chosen = None
    chosen_ctx = None
    for mid in cands:
        mctx = int(_parse_ctx_from_model_id(mid) or 0)
        if mctx <= 0:
            continue
        if want <= mctx and (chosen is None or mctx < chosen_ctx):
            chosen = mid
            chosen_ctx = mctx
    if chosen:
        return chosen
    return max(cands, key=lambda x: int(_parse_ctx_from_model_id(x) or 0))

def _pick_chat_ladder_model_stepup(tokens_needed: int, min_ctx_exclusive: int = 0) -> str:
    """
    Predictive chat escalation helper:
    choose the smallest ladder model that both fits tokens_needed and is strictly
    larger than the current bearer ctx. This prevents 2k -> CPU0 2k migrations.
    """
    want = int(tokens_needed or 0)
    floor_ctx = int(min_ctx_exclusive or 0)
    cands = CHAT_LADDER_MODELS or [DEEP_CPU_MODEL_ID]
    chosen = None
    chosen_ctx = None
    for mid in cands:
        mctx = int(_parse_ctx_from_model_id(mid) or 0)
        if mctx <= floor_ctx:
            continue
        if want <= mctx and (chosen is None or mctx < chosen_ctx):
            chosen = mid
            chosen_ctx = mctx
    if chosen:
        return chosen
    larger = [m for m in cands if int(_parse_ctx_from_model_id(m) or 0) > floor_ctx]
    if larger:
        return min(larger, key=lambda x: int(_parse_ctx_from_model_id(x) or 0))
    return _pick_chat_ladder_model(want)

def _pick_chat_ladder_node(target_model: str) -> str:
    """Pick the chat bearer for ctx escalation with an explicit handover map.

    Policy:
      2K  -> B-PHONE0
      4K  -> B-PHONE3
      8K+ -> B-CPU0
      32K -> B-GPU0 preferred

    Notes:
    - This keeps chat continuity aligned with the intended telecom-style bearer path.
    - We still verify model presence on the candidate node when discovery data is available.
    - Final transport fallback is still handled elsewhere through ROUTER_CHAT_FAILOVER_ORDER,
      so if a preferred bearer is down the router can try the next one.
    """
    model_ctx = int(_parse_ctx_from_model_id(target_model) or 0)

    if model_ctx >= 32768:
        preferred = ["B-GPU0", "B-CPU0", "B-PHONE3", "B-PHONE0"]
    elif model_ctx >= 8192:
        preferred = ["B-CPU0", "B-GPU0", "B-PHONE3", "B-PHONE0"]
    elif model_ctx >= 4096:
        preferred = ["B-PHONE3", "B-CPU0", "B-GPU0", "B-PHONE0"]
    else:
        preferred = ["B-PHONE0", "B-PHONE3", "B-CPU0", "B-GPU0"]

    # Preserve any custom node order while keeping the ctx-tier bearer first.
    configured = list(CHAT_LADDER_NODES) if CHAT_LADDER_NODES else []
    if configured:
        merged = []
        for nn in preferred + configured:
            if nn not in merged:
                merged.append(nn)
        preferred = merged

    for nn in preferred:
        n = _nodes.get(nn)
        if not n:
            continue
        if getattr(n, "models", None) is None or target_model in getattr(n, "models", {}):
            return nn

    return preferred[0] if preferred else "B-CPU0"



def _build_longtext_messages(messages: list[dict], keep_last_n: int = 4, char_budget: int = 12000) -> list[dict]:
    msgs = list(messages or [])
    if not msgs:
        return msgs

    tail = msgs[-max(1, int(keep_last_n or 1)):]
    if len(tail) <= 1:
        only = dict(tail[-1])
        if str(only.get("role") or "") != "user":
            only["role"] = "user"
        return [only]

    doc_msg = dict(tail[-1])
    if str(doc_msg.get("role") or "") != "user":
        doc_msg["role"] = "user"

    leading_systems = []
    history_msgs = []
    seen_non_system = False
    for m in tail[:-1]:
        role = str((m or {}).get("role") or "").lower()
        if role == "system" and not seen_non_system:
            leading_systems.append(m)
        else:
            seen_non_system = True
            if role in ("user", "assistant"):
                history_msgs.append(m)

    history_block = _history_msgs_to_system_block(history_msgs, char_budget=char_budget)

    rebuilt = []
    if leading_systems:
        rebuilt.append(leading_systems[0])
    if history_block:
        rebuilt.append(history_block)
    rebuilt.append(doc_msg)
    return rebuilt


LONGTEXT_TOOL_SUMMARY_MAX_CHARS = int(os.getenv("LONGTEXT_TOOL_SUMMARY_MAX_CHARS", "12000"))
LONGTEXT_TOOL_DOC_MAX_CHARS = int(os.getenv("LONGTEXT_TOOL_DOC_MAX_CHARS", "120000"))
LONGTEXT_TOOL_ACK_BYPASS = int(os.getenv("LONGTEXT_TOOL_ACK_BYPASS", "1"))

_ACK_ONLY_PAT = re.compile(
    r"^\s*(?:"
    r"(?:thanks|thank\s+you|thx|tnx|merci)(?:\s+(?:for|so|very|a\s+lot|lot|the|this|that|your|ur|u|summary|help|response|answer|writeup|write-up|article))*"
    r"|(?:ok(?:ay)?|cool|great|perfect|nice|good|awesome|super)(?:\s+(?:thanks|thank\s+you))?"
    r")"
    r"(?:\s*[:;]-?[)D]|\s*[!.]+)?\s*$",
    re.IGNORECASE,
)


def _is_ack_only_turn(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if len(t) > 120:
        return False
    if bool(_ACK_ONLY_PAT.match(t)):
        return True
    tl = re.sub(r"\s+", " ", t.lower())
    ack_terms = ("thanks", "thank you", "thx", "tnx", "merci")
    if any(term in tl for term in ack_terms):
        disqualifiers = (
            "?", "why", "how", "can you", "could you", "please", "expand", "explain",
            "detail", "elaborate", "continue", "focus", "compare", "what about", "now",
        )
        if not any(d in tl for d in disqualifiers):
            return True
    return False


def _build_ack_only_messages(user_text: str, base_system: str = "") -> list[dict]:
    rebuilt = []
    if str(base_system or "").strip():
        rebuilt.append({"role": "system", "content": str(base_system)})
    rebuilt.append({
        "role": "system",
        "content": "User is acknowledging the previous answer. Reply briefly and politely. Do not continue the previous task unless the user explicitly asks."
    })
    rebuilt.append({"role": "user", "content": str(user_text or "")})
    return rebuilt


def _normalize_text_content_for_router(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and str(p.get("type") or "").lower() in ("text", "input_text"):
                t = p.get("text") or p.get("content") or ""
                if t:
                    parts.append(str(t))
        return "\n".join(parts)
    return str(content or "")


def _build_longtext_tool_messages(messages: list[dict], char_budget: int = 120000) -> list[dict]:
    msgs = list(messages or [])
    if not msgs:
        return msgs

    first_system = None
    for m in msgs:
        if isinstance(m, dict) and str(m.get("role") or "").lower() == "system":
            first_system = dict(m)
            break

    last_user = None
    for m in reversed(msgs):
        if isinstance(m, dict) and str(m.get("role") or "").lower() == "user":
            last_user = dict(m)
            break

    if not last_user:
        return msgs[-1:] if msgs else []

    content = last_user.get("content")
    if isinstance(content, str):
        last_user["content"] = content[-max(1, int(char_budget or 1)):]
    elif isinstance(content, list):
        text_parts = []
        other_parts = []
        for p in content:
            if isinstance(p, dict) and str(p.get("type") or "").lower() in ("text", "input_text"):
                t = p.get("text") or p.get("content") or ""
                if t:
                    text_parts.append(str(t))
            else:
                other_parts.append(p)
        merged = "\n".join(text_parts)[-max(1, int(char_budget or 1)):]
        new_parts = []
        if merged:
            new_parts.append({"type": "text", "text": merged})
        new_parts.extend(other_parts)
        last_user["content"] = new_parts if new_parts else content

    rebuilt = []
    if first_system:
        rebuilt.append(first_system)
    rebuilt.append(last_user)
    return rebuilt


def _build_longtext_followup_messages(summary_text: str, user_text: str, base_system: str = "") -> list[dict]:
    rebuilt = []
    if str(base_system or "").strip():
        rebuilt.append({"role": "system", "content": str(base_system)})
    rebuilt.append({
        "role": "system",
        "content": (
            "Current document summary for this conversation:\n"
            f"{str(summary_text or '').strip()}\n\n"
            "Use this summary as the only document context unless the user pastes a new full article."
        )
    })
    rebuilt.append({"role": "user", "content": str(user_text or "")})
    return rebuilt


def _extract_leading_system_text(messages: list[dict]) -> str:
    for m in messages or []:
        if isinstance(m, dict) and str(m.get("role") or "").lower() == "system":
            return _normalize_text_content_for_router(m.get("content"))
        if isinstance(m, dict) and str(m.get("role") or "").lower() not in ("system",):
            break
    return ""


def _hash_text_for_session(text: str) -> str:
    try:
        return hashlib.sha256(str(text or "").encode("utf-8", "ignore")).hexdigest()
    except Exception:
        return ""


def _extract_assistant_text_from_chat_json_bytes(raw: bytes, max_chars: int = 12000) -> str:
    try:
        j = json.loads((raw or b"").decode("utf-8", "ignore"))
    except Exception:
        return ""
    try:
        choices = j.get("choices") or []
        for ch in reversed(choices):
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content[-max(1, int(max_chars or 1)):]
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        t = p.get("text") or p.get("content") or ""
                        if t:
                            parts.append(str(t))
                out = "\n".join(parts).strip()
                if out:
                    return out[-max(1, int(max_chars or 1)):]
    except Exception:
        return ""
    return ""


def _extract_assistant_text_from_sse_chunks(chunks: list[bytes], max_chars: int = 12000) -> str:
    out_parts = []
    try:
        blob = b"".join(chunks).decode("utf-8", "ignore")
    except Exception:
        return ""
    for line in blob.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
        except Exception:
            continue
        try:
            for ch in (j.get("choices") or []):
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    out_parts.append(delta.get("content") or "")
                elif isinstance(delta.get("content"), list):
                    for p in delta.get("content") or []:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content") or ""
                            if t:
                                out_parts.append(str(t))
                msg = ch.get("message") or {}
                if isinstance(msg.get("content"), str):
                    out_parts.append(msg.get("content") or "")
        except Exception:
            continue
    out = "".join(out_parts).strip()
    return out[-max(1, int(max_chars or 1)):] if out else ""


def _store_longtext_tool_summary(session: Dict[str, Any], summary_text: str, doc_hash: str = "", model_id: str = "") -> None:
    summary = str(summary_text or "").strip()
    if not summary:
        return
    session["longtext_tool_active"] = True
    session["longtext_tool_summary"] = summary[-max(1, int(LONGTEXT_TOOL_SUMMARY_MAX_CHARS or 1)):]
    if doc_hash:
        session["longtext_tool_doc_hash"] = str(doc_hash)
    if model_id:
        session["longtext_tool_summary_model"] = str(model_id)
    session["longtext_tool_updated_at"] = time.time()



def _v98_disable_history_reinject(messages: list[dict]) -> bool:
    try:
        txt = _last_user_text(messages).lower()
    except Exception:
        txt = ""
    markers = ("continue", "follow", "rewrite", "shorten", "summarize", "format", "title", "rephrase")
    return any(m in txt for m in markers)

def _history_msgs_to_system_block(messages: list[dict], char_budget: int = 12000) -> dict | None:
    msgs = list(messages or [])
    if _v98_disable_history_reinject(msgs):
        return None
    if not msgs:
        return None

    budget = int(char_budget or 0)
    lines = []
    for m in reversed(msgs):
        role = str((m or {}).get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue
        content = (m or {}).get("content")
        content_s = content if isinstance(content, str) else str(content or "")
        if not content_s:
            continue

        prefix = "USER" if role == "user" else "ASSISTANT"
        block = f"{prefix}: {content_s}"
        if budget <= 0:
            break
        if len(block) <= budget:
            lines.append(block)
            budget -= len(block)
        else:
            lines.append(block[-budget:])
            budget = 0
            break

    if not lines:
        return None

    lines.reverse()
    return {"role": "system", "content": "Conversation context:\n" + "\n".join(lines)}





_VISION_DESC_PAT = re.compile(
    r"(?:\bdescribe\s+(?:this|the)?\s*image\b|"
    r"\bcan\s+you\s+describe\s+(?:this|the)?\s*image\b|"
    r"\bi\s+need\s+you\s+to\s+help\s*me\s+describe\s+(?:this|the)?\s*image\b|"
    r"\bhelp\s*me\s+describe\s+(?:this|the)?\s*image\b|"
    r"\bwhat\s+is\s+in\s+(?:this|the)\s+image\b|"
    r"\bimage\s+description\b)",
    re.IGNORECASE,
)

def _is_vision_description_prompt(text: str) -> bool:
    if not text:
        return False
    return bool(_VISION_DESC_PAT.search(str(text or "")))



def _current_turn_has_image_payload(messages: list[dict]) -> bool:
    _turn = _current_turn_only_messages(messages)
    return _messages_have_image_payload(_turn)

def _current_turn_only_messages(messages: list[dict]) -> list[dict]:
    msgs = list(messages or [])
    for m in reversed(msgs):
        if isinstance(m, dict):
            return [m]
    return []

def _v98_ui_helper_allowed_for_request(messages: list[dict], body: dict | None = None) -> tuple[bool, str]:
    b = dict(body or {})
    user_text = _last_user_text(messages)
    prompt_chars = len(str(user_text or ""))
    if int(b.get("_scci_intent_coding", 0) or 0) == 1 or int(b.get("_scci_intent_file_code", 0) or 0) == 1:
        return False, "coding_or_file_code"
    if prompt_chars > 600:
        return False, "too_long"
    t = str(user_text or "").lower()
    keywords = (
        "continue","follow","next","rewrite","shorten","summarize","summary",
        "format","fix grammar","improve wording","title","rename","rephrase"
    )
    if not any(k in t for k in keywords):
        return False, "not_micro_helper_task"
    return True, "ok"

def _v98_trim_ui_helper_messages(messages: list[dict], max_chars: int = 600) -> list[dict]:
    msgs = list(messages or [])
    if not msgs:
        return msgs
    tail = []
    for m in reversed(msgs):
        if isinstance(m, dict) and str(m.get("role") or "").lower() in ("user", "assistant", "system"):
            tail.append(dict(m))
            if len(tail) >= 2:
                break
    tail.reverse()
    out = []
    remaining = max(1, int(max_chars or 1))
    for m in tail:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and str(p.get("type") or "").lower() in ("text", "input_text"):
                    t = str(p.get("text") or p.get("content") or "")
                    if t:
                        parts.append(t)
            s = "\n".join(parts)
        else:
            s = str(c or "")
        s = s[:remaining]
        remaining -= len(s)
        out.append({"role": m.get("role"), "content": s})
        if remaining <= 0:
            break
    return out

def _trim_openwebui_helper_messages(messages: list[dict], max_chars: int = 1600, chat_history_max_chars: int = 700) -> list[dict]:
    msgs = list(messages or [])
    if not msgs:
        return msgs
    last_user = None
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = dict(m)
            break
    if not last_user:
        return msgs[-1:] if msgs else msgs

    content = last_user.get("content")
    if isinstance(content, list):
        chunks = []
        for p in content:
            if isinstance(p, dict) and str(p.get("type") or "").lower() in ("text", "input_text"):
                t = p.get("text") or p.get("content") or ""
                if t:
                    chunks.append(str(t))
        s = "\n".join(chunks)
    elif isinstance(content, str):
        s = content
    else:
        s = str(content or "")

    ch_open = s.find("<chat_history>")
    ch_close = s.find("</chat_history>")
    if ch_open != -1 and ch_close != -1 and ch_close > ch_open:
        head = s[: ch_open + len("<chat_history>")]
        hist = s[ch_open + len("<chat_history>") : ch_close]
        tail = s[ch_close:]
        hist = hist[-max(1, int(chat_history_max_chars or 1)):]
        s = head + hist + tail

    s = s[-max(1, int(max_chars or 1)):]
    last_user["content"] = s
    return [last_user]

def _messages_have_image_payload(messages: list[dict]) -> bool:
    msgs = list(messages or [])
    for m in msgs:
        content = (m or {}).get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type") or "").lower()
                if ptype in ("image", "image_url", "input_image"):
                    return True
                if part.get("image_url") or part.get("image") or part.get("url"):
                    return True
        elif isinstance(content, dict):
            ptype = str(content.get("type") or "").lower()
            if ptype in ("image", "image_url", "input_image"):
                return True
            if content.get("image_url") or content.get("image") or content.get("url"):
                return True
        if (m or {}).get("images") or (m or {}).get("files"):
            # Some UIs surface image/file payloads here.
            return True
    return False

def _normalize_user_text_for_coding(text: str) -> str:
    t = str(text or "")
    if not t:
        return ""
    # Strip common OpenWebUI / helper wrappers before classification
    t = re.sub(r"<chat_history>.*?</chat_history>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"###\s*Chat\s+History:.*?(?=###\s*[A-Z][^\n]*:|$)", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"###\s*Guidelines:.*?(?=###\s*[A-Z][^\n]*:|$)", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"###\s*Output:.*?(?=###\s*[A-Z][^\n]*:|$)", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"###\s*Task:\s*(?:Continue|Follow|Generate)\b.*?(?=###\s*[A-Z][^\n]*:|$)", " ", t, flags=re.DOTALL | re.IGNORECASE)
    # Remove generic XML-like wrappers but keep inner text already preserved above
    t = re.sub(r"</?[a-zA-Z_][^>]{0,80}>", " ", t)
    # If fenced code exists, prefer the last fenced block plus a short natural-language prefix
    blocks = _extract_code_blocks(t)
    if blocks:
        prefix = re.sub(r"```(?:[a-zA-Z0-9_+\-]+)?\n?.*?```", " ", t, flags=re.DOTALL).strip()
        prefix = re.sub(r"\s+", " ", prefix).strip()
        if len(prefix) > 240:
            prefix = prefix[:240]
        code = blocks[-1].strip()
        return (prefix + "\n\n```\n" + code + "\n```").strip() if prefix else ("```\n" + code + "\n```")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _detect_code_or_deep(last_user: str) -> bool:
    if not last_user:
        return False

    t = _normalize_user_text_for_coding(last_user)
    tl = t.lower().strip()

    strong_patterns = [
        "```",
        "traceback",
        "stack trace",
        "exception:",
        "syntaxerror",
        "segmentation fault",
        "debug this code",
        "fix this code",
        "fix this:",
        "why does this fail",
        "help me debug this code",
        "python code",
        "javascript code",
        "typescript code",
        "write a python script",
        "compile error",
        "missing colon",
        "indentationerror",
        "unexpected eof",
    ]
    for p in strong_patterns:
        if p in tl:
            return True

    plain_code_pats = [
        r"^\s*for\s+\w+\s+in\s+range\s*\(",
        r"^\s*if\s+.+",
        r"^\s*def\s+\w+\s*\(",
        r"^\s*print\s*\(",
        r"^\s*return\b",
        r"^\s*import\s+\w+",
        r"^\s*from\s+\w+\s+import\b",
    ]
    if len(t) <= 800:
        for pat in plain_code_pats:
            if re.search(pat, t, re.IGNORECASE | re.MULTILINE):
                return True

    weak_hits = 0
    weak_patterns = [
        "def ",
        "class ",
        "import ",
        "from ",
        "console.log",
        "print(",
        "#include",
        "public static void",
        "return ",
        "for ",
        "while ",
        "if ",
        "{",
        "}",
        ";",
    ]
    for p in weak_patterns:
        if p in tl:
            weak_hits += 1

    if weak_hits >= 2:
        return True

    if len(t) <= 400:
        lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
        if lines:
            codeish_lines = 0
            for ln in lines:
                if re.search(r"(\bdef\b|\bfor\b|\bif\b|\bwhile\b|\breturn\b|\bimport\b|\bfrom\b|print\s*\(|[{}();:=])", ln, re.IGNORECASE):
                    codeish_lines += 1
            if codeish_lines >= 1:
                return True

    return False



_CODE_FILE_EXTENSIONS = {
    ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala",
    ".sh", ".bash", ".zsh", ".ps1", ".lua", ".pl", ".r", ".sql", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".scss", ".md", ".dockerfile",
}

_FILE_CODE_INTENT_PAT = re.compile(
    r"(\bexplain\b|\bsummarize\b|\banaly[sz]e\b|\bdebug\b|\bfix\b|\breview\b|\bwhat(?:'s|\s+is)?\s+inside\b|\bunderstand\b|\bwalk\s+me\s+through\b|\bdescribe\b).{0,60}(\bfile\b|\bcode\b|\bscript\b)|"
    r"(\bpython\s+file\b|\bsource\s+code\b|\bthis\s+file\b)",
    re.IGNORECASE | re.DOTALL,
)

def _normalize_possible_filename(value: Any) -> str:
    try:
        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""
        s = s.replace('\\', '/')
        return s.split('/')[-1].strip()
    except Exception:
        return ""

def _looks_like_code_filename(name: str) -> bool:
    base = _normalize_possible_filename(name)
    if not base:
        return False
    low = base.lower()
    if low == 'dockerfile':
        return True
    return any(low.endswith(ext) for ext in _CODE_FILE_EXTENSIONS)

def _extract_filenames_from_obj(obj: Any, out: list[str]) -> None:
    try:
        if obj is None:
            return
        if isinstance(obj, dict):
            for key in ('filename', 'name', 'file_name', 'path', 'basename'):
                val = obj.get(key)
                if isinstance(val, str):
                    nm = _normalize_possible_filename(val)
                    if nm:
                        out.append(nm)
            for key in ('files', 'attachments', 'images', 'items'):
                val = obj.get(key)
                if isinstance(val, list):
                    for it in val:
                        _extract_filenames_from_obj(it, out)
            c = obj.get('content')
            if isinstance(c, list):
                for it in c:
                    _extract_filenames_from_obj(it, out)
        elif isinstance(obj, list):
            for it in obj:
                _extract_filenames_from_obj(it, out)
    except Exception:
        return

def _collect_possible_filenames(body: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[str]:
    out: list[str] = []
    _extract_filenames_from_obj(body, out)
    _extract_filenames_from_obj(messages, out)
    seen: list[str] = []
    for name in out:
        nm = _normalize_possible_filename(name)
        if nm and nm not in seen:
            seen.append(nm)
    return seen

def _detect_attached_code_files(body: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[str]:
    return [nm for nm in _collect_possible_filenames(body, messages) if _looks_like_code_filename(nm)]

def _is_file_code_intent(text: str) -> bool:
    return bool(_FILE_CODE_INTENT_PAT.search(str(text or '')))

def _messages_have_tool_prompt(messages: List[Dict[str, Any]]) -> bool:
    try:
        for m in messages or []:
            if TOOL_SELECT_PAT.search(str((m or {}).get('content', ''))):
                return True
    except Exception:
        pass
    return False

def _winner_from_intent_matrix(flags: Dict[str, bool]) -> tuple[str, list[str]]:
    order = ['vision', 'file_code', 'coding', 'tool', 'ui_helper']
    winner = 'chat'
    for key in order:
        if bool(flags.get(key)):
            winner = key
            break
    overridden = [k for k in order if k != winner and bool(flags.get(k))]
    return winner, overridden

def _scci_log_intent_matrix(req_id: str, flags: Dict[str, bool], winner: str, overridden: list[str], detail: str = '') -> None:
    try:
        log.info(
            f"[{req_id}] → SCCI INTENT_MATRIX sid={req_id} "
            f"vision={1 if flags.get('vision') else 0} file_code={1 if flags.get('file_code') else 0} "
            f"coding={1 if flags.get('coding') else 0} tool={1 if flags.get('tool') else 0} ui_helper={1 if flags.get('ui_helper') else 0} "
            f"winner={winner} override={','.join(overridden) if overridden else '-'}{(' ' + detail) if detail else ''}"
        )
    except Exception:
        pass

def _session_isolation_reset(session: Dict[str, Any], req_id: str, new_lane: str, preserve_longtext_tool: bool = False) -> None:
    try:
        prev = str(session.get('scci_last_lane') or '').strip()
        new_lane_s = str(new_lane or '').strip()
        if prev and prev != new_lane_s:
            if bool(preserve_longtext_tool) and prev == 'longtext_tool':
                log.info(
                    f"[{req_id}] → SCCI SESSION_RESET_SKIP sid={req_id} "
                    f"from_lane={prev} to_lane={new_lane_s} preserve=longtext_tool"
                )
                session['scci_last_lane'] = new_lane_s
                return

            cleared = []
            for k in (
                'medical_demo_active', 'medical_intent', 'last_medical_intent',
                'vision_active', 'last_vision_bucket', 'vision_followup_active',
                'router_medical_demo', 'semantic_lane', 'scci_semantic_lane',
                'longtext_tool_active', 'longtext_tool_summary', 'longtext_tool_doc_hash',
                'longtext_tool_summary_model', 'longtext_tool_updated_at',
            ):
                if k in session:
                    session.pop(k, None)
                    cleared.append(k)
            log.info(
                f"[{req_id}] → SCCI SESSION_RESET sid={req_id} from_lane={prev} to_lane={new_lane_s} "
                f"cleared={','.join(cleared) if cleared else '-'}"
            )
        session['scci_last_lane'] = new_lane_s
    except Exception:
        pass

_OPENWEBUI_INTERNAL_PAT = re.compile(
    r"(###\s*Task:|###\s*Guidelines:|###\s*Output:|###\s*Chat\s+History:|JSON\s+format:|follow_ups|\{\s*\"title\"\s*:|\{\s*\"tags\"\s*:)",
    re.IGNORECASE,
)

_OPENWEBUI_HELPER_INTENT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsuggest\s+3\s*[-–]?\s*5\s+relevant\s+follow[- ]?up\s+questions\b",
        r"\bgenerate\s+a\s+concise,?\s*3\s*[-–]?\s*5\s+word\s+title\b",
        r"\bgenerate\s+1\s*[-–]?\s*3\s+broad\s+tags\b",
        r"\bresponse\s+must\s+be\s+a\s+json\s+array\s+of\s+strings\b",
        r"\bthe\s+output\s+must\s+be\s+a\s+single,?\s+raw\s+json\s+object\b",
        r'\bjson\s+format\s*:\s*\{\s*"follow_ups"',
        r'\bjson\s+format\s*:\s*\{\s*"title"',
        r'\bjson\s+format\s*:\s*\{\s*"tags"',
    ]
]

_OPENWEBUI_LOG_BLOB_PAT = re.compile(
    r"(^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s*\|\s*(?:INFO|WARNING|ERROR)\s*\||"
    r"^INFO:\s+\d+\.\d+\.\d+\.\d+:\d+\s*-\s*\"(?:GET|POST)|"
    r"\bTraceback\s*\(most\s+recent\s+call\s+last\)\b|"
    r"\bOperating\s+System:\s+|\bKernel:\s+Linux\b)",
    re.IGNORECASE | re.MULTILINE,
)

def _classify_openwebui_internal_task(last_user: str) -> dict:
    t = str(last_user or "")
    if not t.strip():
        return {"is_helper": False, "score": 0, "signals": []}

    score = 0
    signals = []
    s = t.strip()

    if s.startswith("### Task:"):
        score += 2
        signals.append("task_header")
    if "### Guidelines:" in t:
        score += 1
        signals.append("guidelines")
    if "### Output:" in t or "JSON format:" in t:
        score += 1
        signals.append("output")
    if "<chat_history>" in t and "</chat_history>" in t:
        score += 2
        signals.append("chat_history_xml")
    elif "### Chat History:" in t:
        score += 1
        signals.append("chat_history_header")

    if _OPENWEBUI_INTERNAL_PAT.search(t):
        score += 1
        signals.append("generic_marker")

    for pat in _OPENWEBUI_HELPER_INTENT_PATTERNS:
        if pat.search(t):
            score += 2
            signals.append("known_helper_intent")
            break

    if _OPENWEBUI_LOG_BLOB_PAT.search(t):
        score -= 3
        signals.append("log_blob")

    try:
        if _is_history_query_wrapper(t):
            score -= 2
            signals.append("history_query_wrapper")
    except Exception:
        pass

    if len(t) >= max(3000, int(LONGTEXT_DIRECT_THRESHOLD or 1500) * 2) and "<chat_history>" not in t:
        score -= 2
        signals.append("large_unwrapped_prompt")

    if len(t) >= max(6000, int(LONGTEXT_DIRECT_THRESHOLD or 1500) * 4):
        score -= 1
        signals.append("very_large_prompt")

    _strong_signature = ("known_helper_intent" in signals) or ("task_header" in signals and ("output" in signals or "chat_history_xml" in signals))
    _looks_like_real_helper = bool(_strong_signature) and not _OPENWEBUI_LOG_BLOB_PAT.search(t)
    return {
        "is_helper": (score >= int(ROUTER_OWUI_HELPER_MIN_SCORE or 5)) and _looks_like_real_helper,
        "score": score,
        "signals": signals,
    }

def _detect_openwebui_internal_task(last_user: str) -> bool:
    return bool(_classify_openwebui_internal_task(last_user).get("is_helper", False))

def _tool_ctx_safe_limit(model_id: str) -> int:
    ctx = int(_parse_ctx_from_model_id(model_id) or 1024)
    safe_ratio = float(ROUTER_TOOL_CTX_SAFE_RATIO or 0.60)
    reserve = int(ROUTER_CTX_COMPLETION_RESERVE or 0)
    return max(128, int(ctx * safe_ratio) - reserve)


_LONGTEXT_HISTORY_WRAPPER_PAT = re.compile(r"^\s*History:\s*.*?\bQuery:\s*", re.IGNORECASE | re.DOTALL)
_LONGTEXT_DOC_MARKERS_PAT = re.compile(r"(\bwikipedia\b|\barticle\b|\bdocument\b|\bchapter\b|\bsection\b|^\s*#\s+|^\s*##\s+|^\s*###\s+|^\s*[-*]\s+)", re.IGNORECASE | re.MULTILINE)

def _is_history_query_wrapper(text: str) -> bool:
    t = str(text or "")
    return bool(_LONGTEXT_HISTORY_WRAPPER_PAT.search(t))

def _is_probable_pasted_document(text: str) -> bool:
    t = str(text or "")
    if not t:
        return False
    if _LONGTEXT_DOC_MARKERS_PAT.search(t):
        return True
    newline_count = t.count("\n")
    sentence_hits = len(re.findall(r"[.!?](?:\s|$)", t))
    if len(t) >= max(2500, int(LONGTEXT_DIRECT_THRESHOLD or 1500) * 2) and (newline_count >= 4 or sentence_hits >= 8):
        return True
    if len(t) >= max(5000, int(LONGTEXT_DIRECT_THRESHOLD or 1500) * 3):
        return True
    return False

def _should_admit_longtext(last_user: str, tokens_needed: int) -> tuple[bool, str]:
    t = str(last_user or "")
    if not t:
        return (False, "empty")
    if int(tokens_needed or 0) >= int(LONGTEXT_EMERGENCY_TOKENS or 12000):
        return (True, "emergency_tokens")
    if len(t) < int(LONGTEXT_DIRECT_THRESHOLD or 1500):
        return (False, "below_threshold")
    if not _is_probable_pasted_document(t):
        return (False, "not_document_like")
    return (True, "document_like")

def _pick_coder_model(tokens_needed: int) -> str:
    # Backward-compatible helper kept for older paths; v9 coding uses
    # _pick_coding_model_by_classification().
    if tokens_needed <= 4096:
        return CODER_MEDIUM_MODEL_4K
    if tokens_needed <= 8192:
        return CODER_MEDIUM_MODEL_8K
    if tokens_needed <= 16384:
        return CODER_MEDIUM_MODEL_16K
    return CODER_MEDIUM_MODEL_32K

def _build_coding_tool_messages(messages: list[dict], char_budget: int = 12000) -> tuple[list[dict], dict]:
    """Build a STRICT coder payload: latest user turn only.

    Coding lane must behave like a focused tool. No prior user/assistant turns,
    no inherited system prompts, and no replay of chat history. The output must
    always be either [last_user] or a 1-message fallback.
    """
    msgs = list(messages or [])
    if not msgs:
        return msgs, {"trimmed": 0, "orig_messages": 0, "new_messages": 0, "orig_chars": 0, "new_chars": 0}

    orig_chars = len(_extract_text_for_estimate(msgs) or "")
    last_user = None
    for m in reversed(msgs):
        if isinstance(m, dict) and str(m.get("role") or "").lower() == "user":
            last_user = dict(m)
            break

    narrowed = [last_user] if last_user else [dict(msgs[-1])]

    content = narrowed[0].get("content")
    if isinstance(content, str):
        cleaned = _normalize_user_text_for_coding(content)
        narrowed[0]["content"] = cleaned[-max(1, int(char_budget or 1)):]
    elif isinstance(content, list):
        text_parts = []
        other_parts = []
        for p in content:
            if isinstance(p, dict) and str(p.get("type") or "").lower() in ("text", "input_text"):
                t = p.get("text") or p.get("content") or ""
                if t:
                    text_parts.append(str(t))
            else:
                other_parts.append(p)
        merged = _normalize_user_text_for_coding("\n".join(text_parts))[-max(1, int(char_budget or 1)):]
        new_parts = []
        if merged:
            new_parts.append({"type": "text", "text": merged})
        new_parts.extend(other_parts)
        narrowed[0]["content"] = new_parts if new_parts else content

    new_chars = len(_extract_text_for_estimate(narrowed) or "")
    return narrowed, {
        "trimmed": 1 if len(msgs) != 1 or new_chars < orig_chars else 0,
        "orig_messages": len(msgs),
        "new_messages": 1,
        "orig_chars": orig_chars,
        "new_chars": new_chars,
    }


def _extract_code_blocks(text: str) -> list[str]:
    try:
        return [m.group(1).strip() for m in re.finditer(r"```(?:[a-zA-Z0-9_+-]+)?\n?(.*?)```", str(text or ""), re.DOTALL)]
    except Exception:
        return []


def _coding_prompt_fingerprint(messages: list[dict]) -> dict:
    last_user = _normalize_user_text_for_coding(_last_user_text(messages))
    blocks = _extract_code_blocks(last_user)
    code_text = "\n\n".join(blocks).strip()
    prompt_text = str(last_user or "")
    if code_text:
        prompt_text = prompt_text.replace(code_text, " ").strip()
    return {
        "last_user": prompt_text.strip(),
        "code_text": code_text,
        "prompt_chars": len(prompt_text or ""),
        "code_chars": len(code_text or ""),
        "has_code_block": 1 if code_text else 0,
        "code_block_count": len(blocks),
    }


def _coding_keyword_score(text: str, patterns: list[str]) -> int:
    score = 0
    t = str(text or "")
    for pat in patterns:
        if re.search(pat, t, re.IGNORECASE):
            score += 1
    return score


def _coding_heuristic_classification(messages: list[dict], file_code_forced: bool = False) -> tuple[str, dict]:
    fp = _coding_prompt_fingerprint(messages)
    last_user = str(fp.get("last_user") or "").strip()
    code_text = str(fp.get("code_text") or "").strip()
    joined = "\n".join([last_user, code_text]).strip()
    low = joined.lower()

    prompt_chars = int(fp.get("prompt_chars") or 0)
    code_chars = int(fp.get("code_chars") or 0)
    code_blocks = int(fp.get("code_block_count") or 0)

    complex_hits = _coding_keyword_score(joined, [
        r"\barchitecture\b", r"\bdistributed\b", r"\bmulti[- ]file\b", r"\brewrite\b",
        r"\brefactor\s+the\s+system\b", r"\bdesign\s+(?:a|the)\s+system\b", r"\bscalable\b",
        r"\bagent\b", r"\bstate machine\b", r"\basyncio\b", r"\bfastapi\b",
        r"\brouter\b", r"\borchestr\w+\b", r"\bcluster\b", r"\bdeep debugging\b",
        r"\bperformance\s+optimization\b", r"\bconcurrency\b", r"\bthread[- ]safe\b"
    ])
    medium_hits = _coding_keyword_score(joined, [
        r"\brefactor\b", r"\badd\s+(?:a|an|the)?\s*feature\b", r"\bextend\b",
        r"\bintegrat\w+\b", r"\bendpoint\b", r"\bapi\b", r"\bclass\b",
        r"\bmodule\b", r"\bfunction\b", r"\bdebug\b", r"\btraceback\b",
        r"\bexception\b", r"\btests?\b", r"\bfix\b", r"\bimprove\b"
    ])
    simple_hits = _coding_keyword_score(joined, [
        r"\bsyntax error\b", r"\btypo\b", r"\bsmall fix\b", r"\bexplain this code\b",
        r"\bwhat does this code do\b", r"\bone function\b", r"\bquick fix\b"
    ])

    trivial_fix = 1 if (
        any(k in low for k in [
            "fix this", "fix this:", "why does this fail", "syntax", "syntax error", "missing",
            "unexpected", "indentation", "missing colon", "missing parenthesis", "quick fix"
        ]) or bool(re.search(r"^\s*(def\s+\w+\s*\(|for\s+\w+\s+in\s+range\s*\(|if\s+.+|print\s*\()", last_user, re.IGNORECASE | re.MULTILINE))
    ) else 0

    architecture_words = 1 if any(k in low for k in ["architecture", "distributed", "system design", "cluster", "orchestrator"]) else 0

    classification = "medium_code"
    reasons = []

    if trivial_fix and architecture_words == 0 and complex_hits == 0:
        if (prompt_chars + code_chars) <= 1800 and code_blocks <= 1:
            classification = "simple_code"
            reasons.append("trivial_fix_guard")

    if classification != "simple_code" and (prompt_chars + code_chars) <= 220 and architecture_words == 0:
        classification = "simple_code"
        reasons.append("tiny_payload_guard")

    if file_code_forced:
        reasons.append("file_code_forced")
        medium_hits += 1

    if classification != "simple_code" and prompt_chars <= int(CODING_SIMPLE_PROMPT_CHAR_MAX or 900) and code_chars <= int(CODING_SIMPLE_CODE_CHAR_MAX or 2200) and code_blocks <= 1 and complex_hits == 0 and medium_hits <= 2:
        classification = "simple_code"
        reasons.append("small_payload")

    if classification != "simple_code":
        if code_chars >= int(CODING_COMPLEX_CODE_CHAR_MIN or 7000) or prompt_chars >= int(CODING_COMPLEX_PROMPT_CHAR_MIN or 2200) or code_blocks >= 3 or complex_hits >= 2 or architecture_words:
            classification = "complex_code"
            reasons.append("large_or_complex_signals")
        elif medium_hits >= 1 or code_blocks >= 1 or file_code_forced:
            classification = "medium_code"
            reasons.append("medium_signals")

    if simple_hits and classification == "medium_code" and complex_hits == 0 and prompt_chars < 700 and code_chars < 1800:
        classification = "simple_code"
        reasons.append("simple_keywords")

    if trivial_fix and architecture_words == 0 and classification == "complex_code":
        classification = "medium_code"
        reasons.append("cap_trivial_not_complex")

    if trivial_fix and architecture_words == 0 and (prompt_chars + code_chars) <= 1800:
        classification = "simple_code"
        if "trivial_fix_guard" not in reasons:
            reasons.append("trivial_fix_final")

    return classification, {
        "prompt_chars": prompt_chars,
        "code_chars": code_chars,
        "code_block_count": code_blocks,
        "complex_hits": complex_hits,
        "medium_hits": medium_hits,
        "simple_hits": simple_hits,
        "trivial_fix": trivial_fix,
        "architecture_words": architecture_words,
        "reasons": reasons,
    }



def _classification_rank(label: str) -> int:
    order = {"simple_code": 0, "medium_code": 1, "complex_code": 2}
    return int(order.get(str(label or "medium_code"), 1))


def _v974_reasoning_level(text: str) -> str:
    t = str(text or "").lower()
    score = 0

    strong_arch = (
        "design", "architecture", "distributed", "scalable", "system",
        "failover", "fallback", "pipeline", "router", "orchestrator",
        "dynamic context escalation", "openai compatibility",
        "failure handling", "multi-node", "cluster"
    )
    strong_reason = (
        "why", "explain", "tradeoff", "trade-off", "compare",
        "analyze", "analyse", "decision", "root cause", "pros", "cons"
    )

    for w in strong_arch:
        if w in t:
            score += 2

    for w in strong_reason:
        if w in t:
            score += 2

    if ("design" in t and "router" in t) or ("design" in t and "system" in t):
        score += 3

    if len(t) > 300:
        score += 1
    if len(t) > 900:
        score += 1

    if score >= 6:
        return "high"
    if score >= 3:
        return "medium"
    return "low"

def _v973_should_promote_simple(user_text: str, class_meta: dict) -> bool:
    meta = dict((class_meta or {}).get("meta") or {})
    low = str(user_text or "").lower()
    if int(meta.get("trivial_fix") or 0) == 1:
        return False
    if int(meta.get("architecture_words") or 0) == 1:
        return True
    if int(meta.get("complex_hits") or 0) >= 1:
        return True
    if re.search(r"\b(endpoint|api|fastapi|upload|json|crud|schema|sqlalchemy|pydantic|route|backend|distributed|router|failover|fallback|pipeline|cluster)\b", low, re.IGNORECASE):
        return True
    if str((class_meta or {}).get("classifier_label") or "") in ("medium_code", "complex_code") and int(meta.get("prompt_chars") or 0) >= 120:
        return True
    return False

async def _v973_select_balanced_model(req_id: str) -> tuple[Optional[str], Optional[str], str]:
    chain = [
        (QWEN35_NODE_PRIMARY, QWEN35_4B_MODEL),
        (QWEN35_NODE_FALLBACK, QWEN35_4B_MODEL),
        (CODER_NODE_PRIMARY, CODER_MEDIUM_MODEL_4K),
        (CODER_NODE_FALLBACK, CODER_MEDIUM_MODEL_4K),
        (CODER_NODE_SIMPLE_PRIMARY, CODER_SIMPLE_MODEL_4K),
    ]
    first_existing = None
    busy_reason = ""
    for idx, (nn, desired_model) in enumerate(chain):
        if not nn:
            continue
        if first_existing is None:
            first_existing = (nn, desired_model)
        n = _nodes.get(nn)
        if not n:
            continue
        try:
            await update_node_metrics(n)
        except Exception:
            pass
        try:
            ok = n.healthy_cached() or await check_node_health(n)
            ready = await check_node_ready(n)
        except Exception:
            ok, ready = False, False
        if not (ok and ready):
            if idx == 0:
                busy_reason = f"{nn.lower()}_unavailable"
            continue
        if idx == 0 and _node_looks_busy(n):
            busy_reason = f"{nn.lower()}_busy"
            continue
        resolved = resolve_model_name_for_node(req_id, n, desired_model, desired_model)
        advertised = getattr(n, "models", None)
        if advertised is not None and resolved not in advertised:
            continue
        reason = "primary" if idx == 0 else (busy_reason or f"fallback_{idx}")
        return nn, resolved, reason
    if first_existing:
        return first_existing[0], first_existing[1], "stale_discovery"
    return None, None, "no_coding_candidate"

async def _v973_select_coding_node_and_model(req_id: str, classification: str, reasoning_level: str, tokens_needed: int) -> tuple[Optional[str], Optional[str], str]:
    cls = str(classification or "medium_code")
    reason = str(reasoning_level or "low")
    if cls == "simple_code":
        return await _select_coding_node_and_model(req_id, cls, tokens_needed)
    if cls == "medium_code" and reason in ("medium", "high"):
        return await _v973_select_balanced_model(req_id)
    if cls == "complex_code" and reason != "high":
        return await _v973_select_balanced_model(req_id)
    return await _select_coding_node_and_model(req_id, cls, tokens_needed)


def _merge_coding_classification(heuristic_label: str, classifier_label: str, heuristic_meta: Optional[dict] = None) -> str:
    """Merge heuristic + classifier with strong anti-over-escalation guards."""
    heuristic_meta = dict(heuristic_meta or {})
    h = str(heuristic_label or "medium_code")
    c = str(classifier_label or h)

    prompt_chars = int(heuristic_meta.get("prompt_chars") or 0)
    code_chars = int(heuristic_meta.get("code_chars") or 0)
    complex_hits = int(heuristic_meta.get("complex_hits") or 0)
    code_blocks = int(heuristic_meta.get("code_block_count") or 0)
    trivial_fix = int(heuristic_meta.get("trivial_fix") or 0)
    architecture_words = int(heuristic_meta.get("architecture_words") or 0)

    total_chars = prompt_chars + code_chars

    if trivial_fix and architecture_words == 0 and total_chars <= 1800:
        return "simple_code"
    if total_chars < 220 and architecture_words == 0:
        return "simple_code"

    if h == "complex_code":
        if trivial_fix and architecture_words == 0:
            return "medium_code"
        return "complex_code"

    if h == "simple_code" and c == "complex_code":
        if architecture_words or complex_hits >= 2 or code_blocks >= 3 or prompt_chars >= int(CODING_COMPLEX_PROMPT_CHAR_MIN or 2200) or code_chars >= int(CODING_COMPLEX_CODE_CHAR_MIN or 7000):
            return "complex_code"
        return "medium_code"

    if h == "medium_code" and c == "complex_code":
        if architecture_words == 0 and complex_hits == 0 and code_blocks <= 1 and prompt_chars < int(CODING_COMPLEX_PROMPT_CHAR_MIN or 2200) and code_chars < int(CODING_COMPLEX_CODE_CHAR_MIN or 7000):
            return "medium_code"

    if c == "complex_code" and total_chars < 400 and architecture_words == 0:
        return h

    return h if _classification_rank(h) >= _classification_rank(c) else c



def _normalize_classifier_label(text: str) -> str:
    raw = str(text or "").strip().lower()
    if raw in ("simple_code", "medium_code", "complex_code"):
        return raw
    m = re.search(r"\b(simple_code|medium_code|complex_code)\b", raw)
    if m:
        return str(m.group(1))
    compact = re.sub(r"[^a-z_]", "", raw)
    if "simple_code" in compact:
        return "simple_code"
    if "medium_code" in compact:
        return "medium_code"
    if "complex_code" in compact:
        return "complex_code"
    return ""


def _build_classifier_payload_text(messages: list[dict]) -> str:
    fp = _coding_prompt_fingerprint(messages)
    user_text = str(fp.get("last_user") or "").strip()
    code_text = str(fp.get("code_text") or "").strip()
    max_chars = max(256, int(CODING_CLASSIFIER_MAX_INPUT_CHARS or 800))

    def _head_tail(s: str, budget: int) -> str:
        s = str(s or "").strip()
        if len(s) <= budget:
            return s
        head = max(80, budget // 2)
        tail = max(80, budget - head - 5)
        return (s[:head] + "\n...\n" + s[-tail:])[:budget]

    user_budget = min(max_chars // 2, 420)
    code_budget = max(120, max_chars - user_budget - 40)
    pieces = []
    if user_text:
        pieces.append("USER_REQUEST:\n" + _head_tail(user_text, user_budget))
    if code_text:
        pieces.append("CODE_SNIPPET:\n" + _head_tail(code_text, code_budget))
    payload_text = "\n\n".join(pieces).strip()
    return payload_text[:max_chars]

def _pick_coding_model_by_classification(classification: str, tokens_needed: int) -> str:
    need = int(tokens_needed or 0)
    cls = str(classification or "medium_code")
    if cls == "simple_code":
        return CODER_SIMPLE_MODEL_4K
    if cls == "complex_code":
        if need <= 8192:
            return CODER_COMPLEX_MODEL_8K
        if need <= 16384:
            return CODER_COMPLEX_MODEL_16K
        return CODER_COMPLEX_MODEL_32K
    if need <= 4096:
        return CODER_MEDIUM_MODEL_4K
    if need <= 8192:
        return CODER_MEDIUM_MODEL_8K
    if need <= 16384:
        return CODER_MEDIUM_MODEL_16K
    return CODER_MEDIUM_MODEL_32K


def _coding_candidate_nodes(classification: str) -> list[str]:
    cls = str(classification or "medium_code")
    if cls == "simple_code":
        return [CODER_NODE_SIMPLE_PRIMARY, CODER_NODE_SIMPLE_FALLBACK, CODER_NODE_PRIMARY, CODER_NODE_FALLBACK]
    return [CODER_NODE_PRIMARY, CODER_NODE_FALLBACK, CODER_NODE_SIMPLE_PRIMARY]


def _node_looks_busy(node) -> bool:
    try:
        q = int(getattr(getattr(node, "metrics", None), "queue_depth", 0) or 0)
        act = int(getattr(getattr(node, "metrics", None), "active_requests", 0) or 0)
        loading = bool(getattr(getattr(node, "metrics", None), "loading", False))
        return loading or q >= int(CODING_GPU_BUSY_QUEUE_DEPTH or 2) or act >= int(CODING_GPU_BUSY_ACTIVE_REQUESTS or 2)
    except Exception:
        return False


async def _classify_coding_complexity(req_id: str, messages: list[dict], file_code_forced: bool = False) -> tuple[str, dict]:
    heuristic_label, heuristic_meta = _coding_heuristic_classification(messages, file_code_forced=file_code_forced)
    payload_text = _build_classifier_payload_text(messages)
    fallback = {
        "source": "heuristic_only",
        "heuristic_label": heuristic_label,
        "classifier_label": heuristic_label,
        "final_label": heuristic_label,
        "meta": heuristic_meta,
    }
    if not payload_text:
        return heuristic_label, fallback
    node = _nodes.get(CODING_CLASSIFIER_NODE)
    if not node:
        return heuristic_label, fallback
    try:
        ok = node.healthy_cached() or await check_node_health(node)
        ready = await check_node_ready(node)
        if not (ok and ready):
            return heuristic_label, fallback
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "Classify the complexity of this coding request.\n\n"
                    "Return ONLY one of these EXACT labels:\n"
                    "simple_code\n"
                    "medium_code\n"
                    "complex_code\n\n"
                    "Rules:\n"
                    "- simple_code: tiny fixes, short explanations, single function edits.\n"
                    "- medium_code: moderate refactors, multiple functions, moderate debugging.\n"
                    "- complex_code: architecture, distributed systems, advanced debugging, or multi-file rewrites.\n\n"
                    "Do not explain. Do not add punctuation. Output one label only."
                ),
            },
            {"role": "user", "content": payload_text},
        ]
        classifier_payload = {
            "model": CODING_CLASSIFIER_MODEL,
            "messages": classifier_messages,
            "stream": False,
            "temperature": CODING_CLASSIFIER_TEMPERATURE,
            "top_k": CODING_CLASSIFIER_TOP_K,
            "top_p": CODING_CLASSIFIER_TOP_P,
            "repeat_penalty": CODING_CLASSIFIER_REPEAT_PENALTY,
            "max_tokens": CODING_CLASSIFIER_MAX_TOKENS,
            "stop": CODING_CLASSIFIER_STOP or ["\\n"]
        }
        headers = {"content-type": "application/json", "x-request-id": req_id, "x-bucket": "coding-classifier"}
        timeout = httpx.Timeout(CODING_CLASSIFIER_TIMEOUT_S, connect=min(1.5, CODING_CLASSIFIER_TIMEOUT_S))
        resp = await client.post(node.url(OPENAI_CHAT_PATH), headers=headers, json=classifier_payload, timeout=timeout)
        if resp.status_code >= 400:
            body = (resp.text or "")[:200].replace("\n", " ")
            return heuristic_label, {**fallback, "source": f"heuristic_http_{resp.status_code}", "error": body}
        data = resp.json()
        raw_text = str((((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
        raw = _normalize_classifier_label(raw_text)
        if raw not in ("simple_code", "medium_code", "complex_code"):
            return heuristic_label, {**fallback, "source": "heuristic_invalid_classifier_output", "error": raw_text[:120]}
        final_label = _merge_coding_classification(heuristic_label, raw, heuristic_meta)
        source = "llm" if raw == heuristic_label else ("hybrid_guarded" if final_label != raw else "hybrid")
        return final_label, {
            "source": source,
            "heuristic_label": heuristic_label,
            "classifier_label": raw,
            "final_label": final_label,
            "meta": heuristic_meta,
        }
    except Exception as e:
        return heuristic_label, {**fallback, "source": f"heuristic_exc_{type(e).__name__}"}


async def _select_coding_node_and_model(req_id: str, classification: str, tokens_needed: int) -> tuple[Optional[str], Optional[str], str]:
    """
    v9.5 strict coding fallback chain.

    Guarantees:
    - simple_code  : PHONE3 1.5B -> CPU0 3B -> GPU0 3B
    - medium_code  : GPU0 3B -> CPU0 3B -> PHONE3 1.5B
    - complex_code : GPU0 DeepSeek -> CPU0 3B -> PHONE3 1.5B

    This preserves coding-lane integrity even when GPU / CPU are degraded.
    It never intentionally falls back to a generic chat model.
    """
    cls = str(classification or "medium_code")
    if cls == "simple_code":
        chain = [
            (CODER_NODE_SIMPLE_PRIMARY, CODER_SIMPLE_MODEL_4K),
            (CODER_NODE_SIMPLE_FALLBACK, CODER_MEDIUM_MODEL_4K),
            (CODER_NODE_PRIMARY, CODER_MEDIUM_MODEL_4K),
        ]
    elif cls == "complex_code":
        chain = [
            (CODER_NODE_PRIMARY, _pick_coding_model_by_classification("complex_code", tokens_needed)),
            (CODER_NODE_FALLBACK, _pick_coding_model_by_classification("medium_code", min(int(tokens_needed or 0), 4096))),
            (CODER_NODE_SIMPLE_PRIMARY, CODER_SIMPLE_MODEL_4K),
        ]
    else:
        chain = [
            (CODER_NODE_PRIMARY, _pick_coding_model_by_classification("medium_code", tokens_needed)),
            (CODER_NODE_FALLBACK, _pick_coding_model_by_classification("medium_code", min(int(tokens_needed or 0), 4096))),
            (CODER_NODE_SIMPLE_PRIMARY, CODER_SIMPLE_MODEL_4K),
        ]

    busy_reason = ""
    first_existing = None

    for idx, (nn, desired_model) in enumerate(chain):
        if not nn:
            continue
        if first_existing is None:
            first_existing = (nn, desired_model)

        n = _nodes.get(nn)
        if not n:
            continue

        try:
            await update_node_metrics(n)
        except Exception:
            pass

        try:
            ok = n.healthy_cached() or await check_node_health(n)
            ready = await check_node_ready(n)
        except Exception:
            ok, ready = False, False

        if not (ok and ready):
            if idx == 0:
                busy_reason = f"{nn.lower()}_unavailable"
            continue

        if idx == 0 and _node_looks_busy(n):
            busy_reason = f"{nn.lower()}_busy"
            continue

        resolved = resolve_model_name_for_node(req_id, n, desired_model, desired_model)
        advertised = getattr(n, "models", None)

        if advertised is not None and resolved not in advertised:
            alt_candidates = []
            if nn == CODER_NODE_SIMPLE_PRIMARY:
                alt_candidates = [CODER_SIMPLE_MODEL_4K, CODER_MEDIUM_MODEL_4K]
            elif nn == CODER_NODE_FALLBACK:
                alt_candidates = [CODER_MEDIUM_MODEL_4K, CODER_SIMPLE_MODEL_4K]
            else:
                alt_candidates = [desired_model, CODER_MEDIUM_MODEL_4K, CODER_SIMPLE_MODEL_4K]

            found = None
            for alt in alt_candidates:
                alt_resolved = resolve_model_name_for_node(req_id, n, alt, alt)
                if advertised is None or alt_resolved in advertised:
                    found = alt_resolved
                    break
            if not found:
                continue
            resolved = found

        reason = "primary" if idx == 0 else (busy_reason or f"fallback_{idx}")
        return nn, resolved, reason

    # Last-resort coding-lane preservation: force the tail of the chain even if discovery is stale.
    if chain:
        forced_node, forced_model = chain[-1]
        if forced_node:
            return forced_node, forced_model, (busy_reason or "forced_coding_last_resort")

    if first_existing:
        return first_existing[0], first_existing[1], (busy_reason or "forced_first_existing")

    return None, None, (busy_reason or "no_candidate")
async def _prewarm_node_model(req_id: str, node_name: str, node, model_id: str):
    """Fire-and-forget prewarm with retry-first policy before GPU fallback.

    Notes:
    - Phone nodes may be asleep and need a short wake-up window.
    - CPU keeps its existing retry-first behavior before GPU fallback.
    - We avoid broad fallback for phone prewarm; we just retry a little longer.
    """
    if not (ROUTER_PREWARM_ENABLED and node and model_id):
        return

    timeout = max(0.5, ROUTER_PREWARM_TIMEOUT_MS / 1000.0)
    node_name_norm = str(node_name or "").strip().upper()
    payload = {"model": model_id, "messages": [{"role": "user", "content": " "}], "stream": False, "max_tokens": 1}

    async def _post_once(target_node):
        async with httpx.AsyncClient(timeout=timeout, headers={"Connection": "close"}) as c:
            await c.post(target_node.url(OPENAI_CHAT_PATH), json=payload)

    def _retry_plan_for(target_name: str):
        if str(target_name or "").upper() == "B-CPU0":
            return [(2, 1.0), (3, 2.0)]
        if str(target_name or "").upper() in ROUTER_PREWARM_PHONE_WAKE_NODES:
            plan = []
            delays = list(ROUTER_PREWARM_PHONE_RETRY_DELAYS_S or [])
            attempts = max(1, int(ROUTER_PREWARM_PHONE_RETRY_ATTEMPTS or 1))
            for idx in range(2, attempts + 1):
                delay = delays[idx - 2] if (idx - 2) < len(delays) else (delays[-1] if delays else 1.5)
                plan.append((idx, float(delay)))
            return plan
        return []

    try:
        ROUTER_STATE.set("PREWARM", node=node_name_norm, event=f"prewarm_start:{model_id}")
    except Exception:
        pass

    try:
        await _post_once(node)
        if ROUTER_PREWARM_LOG:
            log.info(f"[{req_id}] prewarm_ok node={node_name} model={model_id} attempt=1")
        try:
            ROUTER_STATE.set("PREWARM_OK", node=node_name_norm, event=f"prewarm_ok:{model_id}")
        except Exception:
            pass
        return
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
        last_err = e
        try:
            ROUTER_STATE.set("PREWARM_RETRY", node=node_name_norm, event=f"{type(e).__name__}:{model_id}")
        except Exception:
            pass
        if ROUTER_PREWARM_LOG:
            log.warning(
                f"[{req_id}] prewarm_failed node={node_name} model={model_id} "
                f"attempt=1 err={type(e).__name__}:{e} prewarm_retry=1"
            )

        retry_plan = _retry_plan_for(node_name_norm)
        for attempt, delay_s in retry_plan:
            try:
                if ROUTER_PREWARM_LOG:
                    log.info(
                        f"[{req_id}] prewarm_retry node={node_name} model={model_id} "
                        f"attempt={attempt} backoff_s={delay_s}"
                    )
                await asyncio.sleep(delay_s)
                await _post_once(node)
                if ROUTER_PREWARM_LOG:
                    log.info(f"[{req_id}] prewarm_ok node={node_name} model={model_id} attempt={attempt}")
                try:
                    ROUTER_STATE.set("PREWARM_OK", node=node_name_norm, event=f"retry_ok:{model_id}:attempt={attempt}")
                except Exception:
                    pass
                return
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e_retry:
                last_err = e_retry
                try:
                    ROUTER_STATE.set("PREWARM_RETRY", node=node_name_norm, event=f"attempt={attempt}:{type(e_retry).__name__}")
                except Exception:
                    pass
                if ROUTER_PREWARM_LOG:
                    log.warning(
                        f"[{req_id}] prewarm_failed node={node_name} model={model_id} "
                        f"attempt={attempt} err={type(e_retry).__name__}:{e_retry}"
                    )

        try:
            fb_name = "B-GPU0"
            fb_node = NODES.get(fb_name)
            if node_name_norm != "B-CPU0" or fb_node is None:
                try:
                    ROUTER_STATE.set("PREWARM_FAILED", node=node_name_norm, event=f"prewarm_failed:{type(last_err).__name__}")
                except Exception:
                    pass
                return
            if ROUTER_PREWARM_LOG:
                log.warning(
                    f"[{req_id}] prewarm_fallback_start from={node_name} to={fb_name} "
                    f"model={model_id} cause={type(last_err).__name__}"
                )
            try:
                ROUTER_STATE.set("HANDOVER", node=fb_name, event=f"prewarm_fallback_from:{node_name_norm}")
            except Exception:
                pass
            await _post_once(fb_node)
            if ROUTER_PREWARM_LOG:
                log.info(
                    f"[{req_id}] prewarm_fallback_ok node={fb_name} model={model_id} "
                    f"cause={type(last_err).__name__}"
                )
            try:
                ROUTER_STATE.set("PREWARM_OK", node=fb_name, event=f"fallback_ok:{model_id}")
            except Exception:
                pass
        except Exception as e2:
            try:
                ROUTER_STATE.set("PREWARM_FAILED", node=node_name_norm, event=f"fallback_failed:{type(e2).__name__}")
            except Exception:
                pass
            if ROUTER_PREWARM_LOG:
                log.warning(
                    f"[{req_id}] prewarm_fallback_failed node=B-GPU0 model={model_id} "
                    f"err={type(e2).__name__}:{e2}"
                )
    except Exception as e:
        try:
            ROUTER_STATE.set("PREWARM_FAILED", node=node_name_norm, event=f"prewarm_error:{type(e).__name__}")
        except Exception:
            pass
        if ROUTER_PREWARM_LOG:
            log.warning(f"[{req_id}] prewarm_failed node={node_name} model={model_id} err={type(e).__name__}:{e}")
import json
import time
import time
import hmac
import math
import hashlib
import uuid
import logging
import asyncio
import random
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

ROUTER_DEBUG_HEADERS = int(os.getenv("ROUTER_DEBUG_HEADERS", "1"))  # 1=on, 0=off
ROUTER_TOKEN_EST_CHARS_PER_TOKEN = float(os.getenv("ROUTER_TOKEN_EST_CHARS_PER_TOKEN", "4.0"))
ROUTER_LONGPASTE_ENABLE = int(os.getenv("ROUTER_LONGPASTE_ENABLE", "1"))  # 1=on, 0=off
ROUTER_LONGPASTE_CHAR_THRESHOLD = int(os.getenv("ROUTER_LONGPASTE_CHAR_THRESHOLD", "12000"))
ROUTER_LONGPASTE_CHUNK_CHARS = int(os.getenv("ROUTER_LONGPASTE_CHUNK_CHARS", "7000"))
ROUTER_LONGPASTE_MAX_CHUNKS = int(os.getenv("ROUTER_LONGPASTE_MAX_CHUNKS", "12"))




def _message_has_structured_nontext_payload(message: Dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("images") or message.get("files") or message.get("attachments"):
        return True
    c = message.get("content")
    if isinstance(c, list):
        for part in c:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").lower()
                if ptype and ptype not in ("text", "input_text"):
                    return True
                if any(k in part for k in ("image_url", "image", "url", "input_audio", "audio", "file", "attachment", "file_id", "filename")):
                    return True
            elif not isinstance(part, str):
                return True
    elif isinstance(c, dict):
        return True
    return False

def _messages_have_structured_nontext_payload(messages: List[Dict[str, Any]]) -> bool:
    return any(_message_has_structured_nontext_payload(m) for m in (messages or []))

def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for it in content:
            if isinstance(it, dict):
                t = str(it.get("type") or "").lower()
                if t in ("text", "input_text"):
                    v = it.get("text") or it.get("content") or ""
                    if v:
                        chunks.append(str(v))
            elif isinstance(it, str):
                chunks.append(it)
        return "\n".join(chunks).strip()
    return str(content or "").strip()

def sanitize_for_model(model_name: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Never rewrite multimodal / file / audio payloads.
    if _messages_have_structured_nontext_payload(messages):
        return list(messages or [])

    normalized: List[Dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        if not role:
            continue
        content_s = _coerce_text_content(m.get("content"))
        if not content_s and role != "system":
            continue
        m2 = dict(m)
        m2["role"] = role
        m2["content"] = content_s
        normalized.append(m2)

    strict_template = any(x in str(model_name or "").lower() for x in ("gemma", "llama"))
    if strict_template:
        system_blocks = [str(m.get("content") or "").strip() for m in normalized if m.get("role") == "system" and str(m.get("content") or "").strip()]
        if system_blocks:
            system_text = "\n\n".join(system_blocks)
            for m in normalized:
                if m.get("role") == "user":
                    existing = str(m.get("content") or "").strip()
                    m["content"] = (system_text + "\n\n" + existing).strip() if existing else system_text
                    break
        normalized = [m for m in normalized if m.get("role") != "system"]

    cleaned: List[Dict[str, Any]] = []
    for m in normalized:
        if cleaned and cleaned[-1].get("role") == m.get("role"):
            prev = str(cleaned[-1].get("content") or "").strip()
            cur = str(m.get("content") or "").strip()
            if cur:
                cleaned[-1]["content"] = (prev + "\n\n" + cur).strip() if prev else cur
            continue
        cleaned.append(m)

    # Tiny llama.cpp guard for text-only payloads
    cleaned = [m for m in cleaned if str(m.get("content") or "").strip() != "" or m.get("role") == "system"]
    if not cleaned:
        return [{"role": "user", "content": ""}]
    if str(cleaned[0].get("role") or "") != "user":
        cleaned.insert(0, {"role": "user", "content": ""})
    return cleaned

def sanitize_messages_for_text_only(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Backward-compatible alias used by existing send paths.
    return sanitize_for_model("", messages)



import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---- timezone (stdlib) ----
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # type: ignore

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("ROUTER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("lab-router")

logger = log  # alias for legacy references

# ---------------- Router Observability ----------------
ROUTER_OBS_ENABLE = os.getenv("ROUTER_OBS_ENABLE", "1").strip().lower() in {"1","true","yes"}
ROUTER_OBS_LOG_BUFFER_MAX = max(100, int(os.getenv("ROUTER_OBS_LOG_BUFFER_MAX", "800")))
ROUTER_OBS_SSE_HEARTBEAT_S = max(5, int(os.getenv("ROUTER_OBS_SSE_HEARTBEAT_S", "15")))
ROUTER_OBS_INCLUDE_ACCESS_LOGS = os.getenv("ROUTER_OBS_INCLUDE_ACCESS_LOGS", "0").strip().lower() in {"1","true","yes"}
ROUTER_DEMO_MODE = os.getenv("ROUTER_DEMO_MODE", "1").strip().lower() in {"1","true","yes"}
ROUTER_DISCOVERY_INTERVAL_S = max(5, int(os.getenv("ROUTER_DISCOVERY_INTERVAL_S", "20" if ROUTER_DEMO_MODE else "10")))
ROUTER_DISCOVERY_LOG_CHANGES_ONLY = os.getenv("ROUTER_DISCOVERY_LOG_CHANGES_ONLY", "1" if ROUTER_DEMO_MODE else "0").strip().lower() in {"1","true","yes"}
ROUTER_DISCOVERY_PRESENTATION_LOGS = os.getenv("ROUTER_DISCOVERY_PRESENTATION_LOGS", "1" if ROUTER_DEMO_MODE else "0").strip().lower() in {"1","true","yes"}
ROUTER_UI_HELPER_LOGS = os.getenv("ROUTER_UI_HELPER_LOGS", "1").strip().lower() in {"1","true","yes"}
ROUTER_UI_HELPER_ISOLATE_MEDICAL = os.getenv("ROUTER_UI_HELPER_ISOLATE_MEDICAL", "1").strip().lower() in {"1","true","yes"}
ROUTER_TELECOM_CLEAN_MODE = os.getenv("ROUTER_TELECOM_CLEAN_MODE", "1").strip().lower() in {"1","true","yes"}

class _RouterObs:
    def __init__(self):
        self.started_at = time.time()
        self.lock = asyncio.Lock()
        self.requests_total = 0
        self.requests_in_flight = 0
        self.last_latency_ms = 0.0
        self.total_latency_ms = 0.0
        self.last_status_code = 0
        self.last_path = ""
        self.path_counts = Counter()
        self.status_counts = Counter()
        self.bucket_counts = Counter()
        self.node_counts = Counter()
        self.model_counts = Counter()
        self.error_count = 0
        self.sse_clients = 0
        self.log_buffer = deque(maxlen=ROUTER_OBS_LOG_BUFFER_MAX)
        self.log_seq = 0

    async def request_started(self):
        if not ROUTER_OBS_ENABLE:
            return
        async with self.lock:
            self.requests_in_flight += 1

    async def request_finished(self, path: str, status_code: int, latency_ms: float, bucket: str = "", node: str = "", model: str = ""):
        if not ROUTER_OBS_ENABLE:
            return
        async with self.lock:
            self.requests_total += 1
            self.requests_in_flight = max(0, self.requests_in_flight - 1)
            self.last_latency_ms = float(latency_ms or 0.0)
            self.total_latency_ms += float(latency_ms or 0.0)
            self.last_status_code = int(status_code or 0)
            self.last_path = str(path or "")
            self.path_counts[str(path or "")] += 1
            self.status_counts[str(status_code or 0)] += 1
            if bucket:
                self.bucket_counts[str(bucket)] += 1
            if node:
                self.node_counts[str(node)] += 1
            if model:
                self.model_counts[str(model)] += 1
            if int(status_code or 0) >= 400:
                self.error_count += 1

    async def request_failed(self, path: str, latency_ms: float):
        if not ROUTER_OBS_ENABLE:
            return
        async with self.lock:
            self.requests_total += 1
            self.requests_in_flight = max(0, self.requests_in_flight - 1)
            self.last_latency_ms = float(latency_ms or 0.0)
            self.total_latency_ms += float(latency_ms or 0.0)
            self.last_status_code = 500
            self.last_path = str(path or "")
            self.path_counts[str(path or "")] += 1
            self.status_counts["500"] += 1
            self.error_count += 1

    async def log(self, level: str, message: str):
        if not ROUTER_OBS_ENABLE:
            return
        async with self.lock:
            self.log_seq += 1
            self.log_buffer.append({
                "seq": self.log_seq,
                "ts": time.time(),
                "level": str(level or "INFO"),
                "message": str(message or ""),
            })

    async def snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            avg_latency = (self.total_latency_ms / self.requests_total) if self.requests_total else 0.0
            return {
                "started_at": self.started_at,
                "uptime_s": int(max(0, time.time() - self.started_at)),
                "requests_total": self.requests_total,
                "requests_in_flight": self.requests_in_flight,
                "avg_latency_ms": round(avg_latency, 2),
                "last_latency_ms": round(self.last_latency_ms, 2),
                "last_status_code": self.last_status_code,
                "last_path": self.last_path,
                "error_count": self.error_count,
                "sse_clients": self.sse_clients,
                "path_counts": dict(self.path_counts),
                "status_counts": dict(self.status_counts),
                "bucket_counts": dict(self.bucket_counts),
                "node_counts": dict(self.node_counts),
                "top_models": dict(self.model_counts.most_common(12)),
                "log_buffer_size": len(self.log_buffer),
            }

ROUTER_OBS = _RouterObs()

# ---------------- Router Live State ----------------
class _RouterState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.state = "IDLE"
        self.current_node = ""
        self.current_bucket = ""
        self.current_model = ""
        self.last_event = ""
        self.last_request_path = ""
        self.last_request_method = ""
        self.last_update = time.time()

    def _set_unlocked(self, state: str, node: str = "", bucket: str = "", model: str = "", event: str = "", path: str = "", method: str = ""):
        if state:
            self.state = str(state)
        if node:
            self.current_node = str(node)
        if bucket:
            self.current_bucket = str(bucket)
        if model:
            self.current_model = str(model)
        if event:
            self.last_event = str(event)
        if path:
            self.last_request_path = str(path)
        if method:
            self.last_request_method = str(method)
        self.last_update = time.time()

    def set(self, state: str, node: str = "", bucket: str = "", model: str = "", event: str = "", path: str = "", method: str = ""):
        self._set_unlocked(state=state, node=node, bucket=bucket, model=model, event=event, path=path, method=method)

    async def set_async(self, state: str, node: str = "", bucket: str = "", model: str = "", event: str = "", path: str = "", method: str = ""):
        async with self.lock:
            self._set_unlocked(state=state, node=node, bucket=bucket, model=model, event=event, path=path, method=method)

    async def snapshot(self):
        async with self.lock:
            return {
                "state": self.state,
                "current_node": self.current_node,
                "current_bucket": self.current_bucket,
                "current_model": self.current_model,
                "last_event": self.last_event,
                "last_request_path": self.last_request_path,
                "last_request_method": self.last_request_method,
                "last_update": self.last_update,
            }

ROUTER_STATE = _RouterState()

_DISCOVERY_LAST_SUMMARY = ""

def _format_discovery_summary_from_message(message: str) -> str:
    msg = str(message or "").strip()
    if "summary=" in msg:
        msg = msg.split("summary=", 1)[1].strip()
    msg = re.sub(r"\bcycle=\d+\s*", "", msg)
    return msg

def _to_telecom_discovery_message(message: str) -> str:
    msg = _format_discovery_summary_from_message(message)
    m = re.search(r"nodes=(\d+)\s+(.*)$", msg)
    if not m:
        return msg.replace("[DISCOVERY]", "[DISCOVERY:NODES]")
    nodes_count = m.group(1)
    payload = m.group(2).strip()
    states = []
    for part in payload.split(" | "):
        mm = re.match(r"([^:]+):.*?status=([^,|]+)", part)
        if mm:
            states.append(f"{mm.group(1)}={mm.group(2)}")
    compact = " | ".join(states) if states else payload
    return f"[DISCOVERY:NODES] nodes={nodes_count} {compact}"

def _discovery_log(level: str, message: str):
    global _DISCOVERY_LAST_SUMMARY
    raw_msg = str(message or "")
    summary = _format_discovery_summary_from_message(raw_msg)
    if ROUTER_DISCOVERY_LOG_CHANGES_ONLY and summary == _DISCOVERY_LAST_SUMMARY:
        return
    _DISCOVERY_LAST_SUMMARY = summary
    msg = _to_telecom_discovery_message(raw_msg) if ROUTER_TELECOM_CLEAN_MODE else raw_msg
    if ROUTER_DISCOVERY_PRESENTATION_LOGS and not msg.startswith("[DISCOVERY:NODES]"):
        msg = msg.replace("[DISCOVERY]", "[DISCOVERY:NODES]")
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{int((time.time()%1)*1000):03d} | INFO | {msg}"
    if msg.startswith("[DISCOVERY:NODES]"):
        print(_format_discovery_demo_colored(line), flush=True)
        try:
            _router_obs_safe_create_task(ROUTER_OBS.log("INFO", msg))
        except Exception:
            pass
        return
    level = str(level or "INFO").upper()
    if level == "WARNING":
        log.warning(msg)
    elif level == "ERROR":
        log.error(msg)
    else:
        log.info(msg)


def _router_obs_safe_create_task(coro):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except Exception:
        pass

class _RouterObsLogHandler(logging.Handler):
    def emit(self, record):
        try:
            name = str(getattr(record, "name", "") or "")
            if (not ROUTER_OBS_INCLUDE_ACCESS_LOGS) and name.startswith("uvicorn"):
                return
            msg = self.format(record)
            _router_obs_safe_create_task(ROUTER_OBS.log(record.levelname, msg))
        except Exception:
            pass

ROUTER_COLOR_LOGS = os.getenv("ROUTER_COLOR_LOGS", "1").strip().lower() in {"1","true","yes"}

class _RouterLogColor:
    RESET = "[0m"
    BLUE = "[94m"
    PURPLE = "[95m"
    GREEN = "[92m"
    YELLOW = "[93m"
    RED = "[91m"
    CYAN = "[96m"



def _log_ui_helper(req_id: str, message: str):
    if not ROUTER_UI_HELPER_LOGS:
        return
    try:
        log.info(f"[{req_id}] [UI_HELPER] {message}")
    except Exception:
        pass

def _discovery_segment_color(segment: str) -> str:
    s = str(segment or "").lower()
    if "=ok" in s or "status=ok" in s:
        return _RouterLogColor.GREEN
    if "=cooldown" in s or "cooldown" in s or "retry" in s or "backoff" in s:
        return _RouterLogColor.YELLOW
    if "=fail" in s or "=down" in s or "connecterror" in s or "timeout" in s or "unavailable" in s:
        return _RouterLogColor.RED
    return _RouterLogColor.CYAN

def _format_discovery_demo_colored(full_line: str) -> str:
    if not ROUTER_COLOR_LOGS:
        return full_line
    line = str(full_line or "")
    marker = " | INFO | [DISCOVERY:NODES] "
    idx = line.find(marker)
    if idx == -1:
        return line
    prefix = line[:idx + len(" | INFO | ")]
    rest = line[idx + len(" | INFO | "):]
    header_idx = rest.find("nodes=")
    if header_idx == -1:
        return line
    payload_start = rest.find(" ", header_idx)
    if payload_start == -1:
        return line
    header = rest[:payload_start + 1]
    payload = rest[payload_start + 1:]
    parts = payload.split(" | ")
    pieces = []
    for i, part in enumerate(parts):
        color = _discovery_segment_color(part)
        pieces.append(f"{color}{part}{_RouterLogColor.RESET}")
        if i < len(parts) - 1:
            pieces.append(" | ")
    return prefix + _RouterLogColor.CYAN + header + _RouterLogColor.RESET + "".join(pieces)

def _router_pick_log_color(levelname: str, message: str) -> str:
    """
    Router log color policy (v10.1.0)

    Semantic mapping:
      - RED    : errors / failed fallback / safety or degraded conditions
      - YELLOW : warnings / retries / skips / backoff
      - CYAN   : direct-response / system-knowledge / memory-gate / local answers
      - PURPLE : prewarm / handover / session-state / bearer-state / timeline / media / voice
      - BLUE   : routing decisions / cluster / intent / model selection / capability resolution
      - GREEN  : success / completion / ready / ok states
    """
    m = str(message or "")
    ml = m.lower()
    lvl = str(levelname or "").upper()

    # 1) Error / safety path
    if (
        lvl == "ERROR"
        or "error" in ml
        or "failed" in ml
        or "fallback_failed" in ml
        or "prewarm_failed" in ml
        or "connecttimeout" in ml
        or "degraded_mode" in ml
        or "longtext_ctx_safety" in ml
        or "safety" in ml
    ):
        return _RouterLogColor.RED

    # 2) Warning / retry / trimmed path
    if (
        lvl == "WARNING"
        or "warning" in ml
        or "retry" in ml
        or "backoff" in ml
        or "fallback_start" in ml
        or "longtext_skip" in ml
        or "trim" in ml
        or "queue" in ml
    ):
        return _RouterLogColor.YELLOW

    # 3) Direct local knowledge / memory gate / direct answer path
    if (
        "sys_knowledge" in ml
        or "direct_response" in ml
        or "memory_gate" in ml
        or "[mem_gate]" in ml
        or "router_notice" in ml
        or "aq" in ml
    ):
        return _RouterLogColor.CYAN

    # 4) Stateful transitions / media paths
    if (
        "prewarm" in ml
        or "handover" in ml
        or "scci state" in ml
        or "scci session" in ml
        or "scci bearer_state" in ml
        or "scci timeline" in ml
        or "[media]" in ml
        or "[voice]" in ml
        or "image_generation" in ml
        or "tts" in ml
    ):
        return _RouterLogColor.PURPLE

    # 5) Routing / selection / cluster view
    if (
        "scci decision" in ml
        or "→ route" in ml
        or "model_resolve" in ml
        or "scci intent" in ml
        or "scci cluster" in ml
        or "chat_ladder" in ml
        or "final_rebuild_ctx" in ml
        or "bucket=" in ml
        or "selected node" in ml
        or "capability_match" in ml
    ):
        return _RouterLogColor.BLUE

    # 6) Success / completion / ready
    if (
        "prewarm_ok" in ml
        or "proc_end" in ml
        or "done total_ms=" in ml
        or "completed" in ml
        or "success" in ml
        or "ready" in ml
        or "ok" in ml
    ):
        return _RouterLogColor.GREEN

    return ""

class _RouterColorFormatter(logging.Formatter):
    def format(self, record):
        s = super().format(record)
        if not ROUTER_COLOR_LOGS:
            return s
        color = _router_pick_log_color(getattr(record, "levelname", ""), getattr(record, "msg", ""))
        if not color:
            return s
        return f"{color}{s}{_RouterLogColor.RESET}"

if ROUTER_COLOR_LOGS:
    try:
        _fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s")
        for _h in logging.getLogger().handlers:
            _h.setFormatter(_RouterColorFormatter(_fmt._fmt))
        for _h in log.handlers:
            _h.setFormatter(_RouterColorFormatter(_fmt._fmt))
    except Exception:
        pass

# Avoid noisy httpx/httpcore logs unless you explicitly enable them
if os.getenv("ROUTER_HTTPX_DEBUG", "0").strip() not in ("1", "true", "True", "yes", "YES"):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

if ROUTER_OBS_ENABLE:
    try:
        _router_obs_handler = _RouterObsLogHandler()
        _router_obs_handler.setLevel(logging.INFO)
        _router_obs_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(_router_obs_handler)
    except Exception:
        pass

# ---------------- Request debug logging ----------------
# 0=off, 1=last_user head, 2=roles only, 3=full sanitized (truncated)
ROUTER_LOG_REQUESTS = os.getenv("ROUTER_LOG_REQUESTS", "0").strip()
ROUTER_LOG_IMAGE_DETECT = int(os.getenv("ROUTER_LOG_IMAGE_DETECT", "1"))  # 1=log image detection (default on)
ROUTER_VISION_OVERRIDE_ON_IMAGE = int(os.getenv("ROUTER_VISION_OVERRIDE_ON_IMAGE", "1"))  # 1=force vision bucket when images present
ROUTER_VISION_TEXT_INTENT = int(os.getenv("ROUTER_VISION_TEXT_INTENT", "1"))  # 1=promote likely image-description prompts when image payload exists
ROUTER_SICC_VISION_PAGING = int(os.getenv("ROUTER_SICC_VISION_PAGING", "0"))  # 1=fire-and-forget warmup ping to vision node
ROUTER_LOG_REQUESTS_MAXCHARS = int(os.getenv("ROUTER_LOG_REQUESTS_MAXCHARS", "8000"))
ROUTER_LOG_REQUESTS_LASTUSER_MAXCHARS = int(os.getenv("ROUTER_LOG_REQUESTS_LASTUSER_MAXCHARS", "1200"))

# ---------------- Vision image preprocessing ----------------
ROUTER_VISION_PREPROCESS_ENABLE = int(os.getenv("ROUTER_VISION_PREPROCESS_ENABLE", "1"))
ROUTER_VISION_PREPROCESS_SIZE = int(os.getenv("ROUTER_VISION_PREPROCESS_SIZE", "896"))
ROUTER_VISION_PREPROCESS_BG = os.getenv("ROUTER_VISION_PREPROCESS_BG", "black").strip().lower()
ROUTER_VISION_PREPROCESS_JPEG_QUALITY = int(os.getenv("ROUTER_VISION_PREPROCESS_JPEG_QUALITY", "92"))

# ---------------- Image generation ----------------
IMAGE_GEN_ENABLE = int(os.getenv("ROUTER_IMAGE_GEN_ENABLE", "1"))
# Legacy/default GPU image engine (kept for backward compatibility)
IMAGE_GEN_BACKEND_URL = os.getenv("ROUTER_IMAGE_GEN_BACKEND_URL", "http://192.168.1.62:7860").rstrip("/")
IMAGE_GEN_ENDPOINT = os.getenv("ROUTER_IMAGE_GEN_ENDPOINT", "/sdapi/v1/txt2img").strip()
IMAGE_GEN_NODE_NAME = os.getenv("ROUTER_IMAGE_GEN_NODE_NAME", "B-GPU0-image-engine").strip()
IMAGE_GEN_WIDTH = int(os.getenv("ROUTER_IMAGE_GEN_WIDTH", "512"))
IMAGE_GEN_HEIGHT = int(os.getenv("ROUTER_IMAGE_GEN_HEIGHT", "512"))
IMAGE_GEN_STEPS = int(os.getenv("ROUTER_IMAGE_GEN_STEPS", "20"))
IMAGE_GEN_SAMPLER = os.getenv("ROUTER_IMAGE_GEN_SAMPLER", "Euler").strip()
IMAGE_GEN_OUTPUT_DIR = os.getenv("ROUTER_IMAGE_GEN_OUTPUT_DIR", "/home/mbenslaiman/Brain-Ai/Cognitive_Router/generated_images").strip()
IMAGE_GEN_PUBLIC_BASE = os.getenv("ROUTER_IMAGE_GEN_PUBLIC_BASE", "http://192.168.1.61:8000").rstrip("/")

# Multi-backend image generation lane
IMAGE_GEN_BACKEND_ORDER = [n.strip() for n in os.getenv("ROUTER_IMAGE_GEN_BACKEND_ORDER", "B-PHONE3-image-engine,B-GPU0-image-engine").split(",") if n.strip()]

# Native stable-diffusion.cpp API on B-PHONE3
IMAGE_GEN_BPHONE3_URL = os.getenv("ROUTER_IMAGE_GEN_BPHONE3_URL", "http://192.168.1.66:8080").rstrip("/")
IMAGE_GEN_BPHONE3_ENDPOINT = os.getenv("ROUTER_IMAGE_GEN_BPHONE3_ENDPOINT", "/sdapi/v1/txt2img").strip()
IMAGE_GEN_BPHONE3_NODE_NAME = os.getenv("ROUTER_IMAGE_GEN_BPHONE3_NODE_NAME", "B-PHONE3-image-engine").strip()
IMAGE_GEN_BPHONE3_STEPS = int(os.getenv("ROUTER_IMAGE_GEN_BPHONE3_STEPS", "1"))
IMAGE_GEN_BPHONE3_CFG_SCALE = float(os.getenv("ROUTER_IMAGE_GEN_BPHONE3_CFG_SCALE", "1.0"))
IMAGE_GEN_BPHONE3_SAMPLE_METHOD = os.getenv("ROUTER_IMAGE_GEN_BPHONE3_SAMPLE_METHOD", "euler_a").strip()

# Existing GPU image engine fallback (A1111/WebUI-style)
IMAGE_GEN_GPU_URL = os.getenv("ROUTER_IMAGE_GEN_GPU_URL", IMAGE_GEN_BACKEND_URL).rstrip("/")
IMAGE_GEN_GPU_ENDPOINT = os.getenv("ROUTER_IMAGE_GEN_GPU_ENDPOINT", IMAGE_GEN_ENDPOINT).strip()
IMAGE_GEN_GPU_NODE_NAME = os.getenv("ROUTER_IMAGE_GEN_GPU_NODE_NAME", IMAGE_GEN_NODE_NAME).strip()
IMAGE_GEN_GPU_STEPS = int(os.getenv("ROUTER_IMAGE_GEN_GPU_STEPS", str(IMAGE_GEN_STEPS)))
IMAGE_GEN_GPU_SAMPLER = os.getenv("ROUTER_IMAGE_GEN_GPU_SAMPLER", IMAGE_GEN_SAMPLER).strip()
ARCH_DIAGRAM_ENABLE = int(os.getenv("ROUTER_ARCH_DIAGRAM_ENABLE", "1"))
ARCH_DIAGRAM_FILENAME = os.getenv("ROUTER_ARCH_DIAGRAM_FILENAME", "lab_router_architecture.png").strip()

IMAGE_BACKEND_HEALTH_TIMEOUT_S = float(os.getenv("ROUTER_IMAGE_BACKEND_HEALTH_TIMEOUT_S", "2.5"))

def _image_backend_health_endpoint_candidates(spec: dict) -> list[str]:
    base = str(spec.get("url") or "").rstrip("/")
    return [
        base + "/sdapi/v1/progress",
        base + "/sdapi/v1/samplers",
        base + "/docs",
        base + "/openapi.json",
        base + "/",
    ]

async def _image_backend_is_up(spec: dict) -> bool:
    timeout = max(1.0, float(IMAGE_BACKEND_HEALTH_TIMEOUT_S or 2.5))
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in _image_backend_health_endpoint_candidates(spec):
            try:
                r = await client.get(url)
                if r.status_code < 500:
                    return True
            except Exception:
                continue
    return False

_IMAGE_GEN_PAT = re.compile(
    r"(?:\b(generate|create|draw|make|render)\b.{0,40}\b(image|picture|illustration|artwork|art)\b|\bimage\s+generation\b|\btxt2img\b)",
    re.IGNORECASE | re.DOTALL,
)

def _detect_image_generation_intent(last_user: str) -> bool:
    if not IMAGE_GEN_ENABLE:
        return False
    t = str(last_user or "").strip()
    if not t:
        return False
    return bool(_IMAGE_GEN_PAT.search(t))

_ARCH_DIAGRAM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bshow\s+me\s+the\s+architecture\b",
        r"\bshow\s+the\s+architecture\b",
        r"\blab\s+router\s+architecture\b",
        r"\bshow\s+me\s+the\s+lab\s+router\b",
        r"\bdisplay\s+the\s+architecture\b",
        r"\bshow\s+the\s+router\s+diagram\b",
        r"\bshow\s+me\s+the\s+diagram\b",
        r"\bcontrol\s+plane\s+architecture\b",
        r"\bcan\s+you\s+show\s+me\s+the\s+architecture\s+lab\s+router\b",
        r"\bcan\s+you\s+show\s+me\s+the\s+lab\s+router\s+architecture\b",
    ]
]

def _detect_architecture_diagram_intent(last_user: str) -> bool:
    if not ARCH_DIAGRAM_ENABLE:
        return False
    t = str(last_user or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _ARCH_DIAGRAM_PATTERNS)

_SENSITIVE_KEYS = {
    "authorization", "api_key", "apikey", "token", "access_token", "refresh_token",
    "password", "secret", "cookie", "set-cookie"
}

def _sanitize_obj(obj: Any) -> Any:
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in _SENSITIVE_KEYS:
                    out[k] = "***REDACTED***"
                else:
                    out[k] = _sanitize_obj(v)
            return out
        if isinstance(obj, list):
            return [_sanitize_obj(x) for x in obj]
        return obj
    except Exception:
        return "***UNSAFE_TO_LOG***"

def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            t = (part.get("type") or "").lower()
            if t in ("text", "input_text"):
                chunks.append(part.get("text", ""))
        return "\n".join([c for c in chunks if c])
    return str(content)

def _debug_log_request(body: Dict[str, Any]) -> None:
    lvl = ROUTER_LOG_REQUESTS
    if lvl in ("0", "", "false", "False", "no", "NO"):
        return

    msgs = body.get("messages", [])
    roles: List[Any] = []
    last_user = ""

    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                roles.append(m.get("role"))
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = _extract_text_from_content(m.get("content", ""))
                break

    if lvl == "2":
        log.info("[reqdbg] model=%s stream=%s messages=%d roles=%s",
                 body.get("model"), bool(body.get("stream", False)),
                 len(msgs) if isinstance(msgs, list) else -1, roles)
        return

    if lvl == "3":
        safe = _sanitize_obj(body)
        s = json.dumps(safe, ensure_ascii=False)
        if len(s) > ROUTER_LOG_REQUESTS_MAXCHARS:
            s = s[:ROUTER_LOG_REQUESTS_MAXCHARS] + "…(truncated)"
        log.info("[reqdbg] body=%s", s)
        return

    head = (last_user or "")[:ROUTER_LOG_REQUESTS_LASTUSER_MAXCHARS]
    log.info("[reqdbg] model=%s stream=%s messages=%d roles=%s last_user_len=%d last_user_head=%r",
             body.get("model"), bool(body.get("stream", False)),
             len(msgs) if isinstance(msgs, list) else -1, roles,
             len(last_user or ""), head)


# ---------------- SCCI System Knowledge ----------------
SCCI_SYSTEM_KNOWLEDGE_ENABLE = int(os.getenv("SCCI_SYSTEM_KNOWLEDGE_ENABLE", "1"))
SCCI_SYSTEM_KNOWLEDGE_LOG = int(os.getenv("SCCI_SYSTEM_KNOWLEDGE_LOG", "1"))

SCCI_SYSTEM_QA = [
    {
        "intent": "presentation",
        "patterns": [
            r"\bcan\s+you\s+present\s+yourself\b",
            r"\bpresent\s+yourself\b",
            r"\bintroduce\s+yourself\b",
            r"\bplease\s+introduce\s+yourself\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bexplain\s+in\s+details\s+this\s+lab\s+router\b",
            r"\bexplain\s+this\s+lab\s+router\b",
        ],
        "answer": "I am the SCCI Router — a telecom-inspired cognitive control plane for AI workloads. I analyze each request, classify it by intent, and route it to the appropriate model and node.  I start with lightweight devices and escalate only when necessary."
    },
    {
        "intent": "creator",
        "patterns": [
            r"\bwho\s+created\s+you\b",
            r"\bwho\s+built\s+you\b",
            r"\bwho\s+made\s+you\b",
            r"\bwho\s+developed\s+you\b",
            r"\bwho\s+created\s+this\s+system\b",
            r"\bwho\s+built\s+this\s+system\b",
            r"\bwho\s+designed\s+this\s+system\b",
        ],
        "answer": "I am part of the SCCI Router environment, designed and authored by M.B.E.F from Benslaiman.com."
    },
    {
        "intent": "identity",
        "patterns": [
            r"\bwho\s+are\s+you\b",
            r"\bwhat\s+are\s+you\b",
            r"\bare\s+you\s+chatgpt\b",
            r"\bare\s+you\s+a\s+chatbot\b",
        ],
        "answer": "I am the assistant operating inside the SCCI LAB Router demonstration environment, a telecom-inspired cognitive routing system for local AI clusters."
    },
    {
    "intent": "routing_order",
    "patterns": [
        r"\bhow\s+does\s+the\s+router\s+process\s+a\s+request\b",
        r"\bhow\s+does\s+the\s+router\s+route\s+a\s+request\b",
        r"\bwhat\s+checks\s+does\s+the\s+router\s+perform\b",
        r"\bdo\s+you\s+check\s+internal\s+answers\s+first\b",
        r"\bdo\s+you\s+check\s+faq\s+before\s+the\s+model\b"
    ],
    "answer": "When a request arrives, the router can first answer a small set of deterministic control-plane questions directly, such as system knowledge, operational status, or specific demo controls. If no direct response applies, it evaluates the request through an intent matrix and assigns it to the most appropriate lane, such as vision, coding, tools, UI helper, or general chat. The router then selects the smallest suitable model and node for that lane, while still allowing escalation when the workload, context size, or hardware state requires it."
    },
    {
        "intent": "lab_router",
        "patterns": [
            r"\bwhat\s+is\s+lab\s+router\b",
            r"\bwhat\s+is\s+the\s+lab\s+router\b",
            r"\bwhat\s+is\s+this\s+router\b",
            r"\bwhat\s+is\s+this\s+system\b",
        ],
        "answer": "The SCCI LAB Router is a telecom-inspired cognitive routing engine that orchestrates AI workloads across a cluster of local models."
    },
    {
        "intent": "scci",
        "patterns": [
            r"\bwhat\s+is\s+scci\b",
            r"\bwhat\s+does\s+scci\s+mean\b",
            r"\bdefine\s+scci\b",
        ],
        "answer": "SCCI stands for Smart Cognitive Cluster Intelligence. It represents the decision layer of the LAB Router, responsible for classifying incoming requests and routing them to the most appropriate model and compute node within the cluster. Its goal is to coordinate specialized inference nodes and select the best resource for each workload."
    },
    {
        "intent": "purpose",
        "patterns": [
            r"\bwhat\s+is\s+the\s+purpose\s+of\s+this\s+project\b",
            r"\bwhy\s+did\s+you\s+build\s+this\b",
            r"\bwhat\s+problem\s+does\s+this\s+solve\b",
        ],
        "answer": "This project explores how telecom-inspired control principles can be applied to AI inference, allowing specialized local models to work together as a coordinated system."
    },
    {
        "intent": "telecom_inspiration",
        "patterns": [
            r"\bwhy\s+is\s+this\s+telecom\s+inspired\b",
            r"\bwhy\s+telecom\s+inspired\b",
            r"\bhow\s+is\s+this\s+telecom\s+inspired\b",
        ],
        "answer": "It is telecom-inspired because it applies ideas such as admission, routing, mobility, and continuity to AI workloads instead of radio sessions."
    },
    {
        "intent": "hardware_philosophy",
        "patterns": [
            r"\bwhy\s+did\s+you\s+use\s+modest\s+hardware\b",
            r"\bwhy\s+is\s+the\s+system\s+designed\s+for\b\s+modest\b\s+hardware\b",
            r"\bwhy\s+modest\s+hardware\b",
            r"\bwhy\s+old\s+phones\b",
            r"\bwhy\s+use\s+older\s+phones\b",
            r"\bwhy\s+use\s+a\s+laptop\b",
            r"\bwhy\s+lenovo\b",
            r"\bwhy\s+is\s+the\s+system\s+designed\s+to\s+run\s+on\s+modest\s+hardware\b",
            r"\bwhy\s+does\s+the\s+system\s+use\s+modest\s+hardware\b",
            r"\bwhy\s+do\s+you\s+run\s+on\s+older\s+devices\b",
            r"\bwhy\s+do\s+you\s+use\s+old\s+phones\b",
            r"\bwhy\s+do\s+you\s+use\s+low\s+power\s+devices\b",
            r"\bwhy\s+is\s+the\s+cluster\s+based\s+on\s+modest\s+hardware\b",
            r"\bwhy\s+does\s+the\s+router\s+run\s+on\s+modest\s+hardware\b",
        ],
        "answer": "This prototype explores how AI inference can be orchestrated across modest and heterogeneous hardware. The cluster includes devices such as older mobile phones alongside a ThinkPad T400 laptop acting as the central orchestrator. The goal is to demonstrate that useful AI workloads can run on energy-efficient devices when requests are routed intelligently. Lightweight tasks can be handled by low-power nodes, while more demanding workloads can be escalated to stronger compute resources such as GPU acceleration only when necessary. This approach helps reduce energy usage while still enabling scalable AI inference."
    },
    {
    "intent": "owner_spoken_intro",
    "patterns": [
        r"\bpresent\s+your\s+owner\b",
        r"\bintroduce\s+your\s+owner\b",
        r"\bpresent\s+your\s+creator\b",
        r"\bintroduce\s+your\s+creator\b",
        r"\bpresent\s+the\s+owner\b",
        r"\bintroduce\s+the\s+owner\b",
        r"\bpresent\s+the\s+creator\b",
        r"\bintroduce\s+the\s+creator\b",
        r"\bpresent\s+benslaiman.com\b",
        r"\bintroduce\s+benslaiman.com\b",
        r"\btalk\s+like\s+i(?:'m| am)\s+speaking\b",
        r"\bspeak\s+like\s+i(?:'m| am)\s+speaking\b",
        r"\bwrite\s+it\s+like\s+i(?:'m| am)\s+speaking\b",
        r"\bmake\s+it\s+sound\s+like\s+i(?:'m| am)\s+speaking\b",
        r"\bwrite\s+in\s+first\s+person\b",
        r"\bsay\s+it\s+in\s+first\s+person\b",
        r"\bmake\s+the\s+router\s+present\s+its\s+owner\b",
        r"\bmake\s+the\s+router\s+introduce\s+its\s+creator\b",
        r"\bmake\s+it\s+sound\s+like\s+the\s+creator\s+is\s+speaking\b",
        r"\bmake\s+it\s+sound\s+like\s+benslaiman.com\s+is\s+speaking\b"
        r"\bpresent\s+yourself\s+as\s+your\s+creator\b"
    ],
    "answer": "Hello, my name is M.B.E.F from Benslaiman.com. My background is in telecommunications engineering, where I worked on technologies such as MSS,IMS, VoWiFi, VoLTE, VoNR,and SRVCC, with a focus on mobility and session continuity across networks. Inspired by telecom control-plane architectures, I started exploring how similar principles could be applied to AI infrastructure orchestration. The system I am presenting today is called the SCCI LAB Router. It is an experimental control-plane architecture designed to dynamically orchestrate multiple AI models and compute nodes depending on the task. Instead of relying on a single large model, the LAB Router analyzes each request and selects the most appropriate model and hardware resource for the workload. The cluster used in this demonstration intentionally runs on modest hardware, including older phones and a laptop. The goal is to demonstrate how intelligent orchestration can maximize limited resources through adaptive model selection and workload routing."
    },
    {
        "intent": "image_failover",
        "patterns": [
            r"\bwhat\s+happens\s+if\s+b[- ]?phone3\s+goes\s+offline\b",
            r"\bwhat\s+happens\s+if\s+b[- ]?phone3\s+is\s+down\b",
            r"\bhow\s+does\s+image\s+failover\s+work\b",
            r"\bhow\s+does\s+the\s+router\s+handle\s+image\s+backend\s+failure\b",
            r"\bwhat\s+happens\s+when\s+the\s+primary\s+image\s+backend\s+is\s+unreachable\b"
        ],
        "answer": "For image generation, the router uses an ordered backend list. It tries B-PHONE3-image-engine first and B-GPU0-image-engine second. The router does not assume a node is down in advance. Instead, on the next image-generation request it probes backend availability, and if the primary image backend is unreachable it reroutes the request to the next available backend."
    },
    {
        "intent": "failover_demo_control",
        "patterns": [
            r"\bnow\s+i(?:'| a)?m\s+going\s+to\s+put\s+(?:my\s+)?b[- ]?phone[- ]?3\s+on\s+airplane\s+mode\b",
            r"\bput\s+(?:my\s+)?b[- ]?phone[- ]?3\s+on\s+airplane\s+mode\b",
            r"\bprepare\s+(?:the\s+router\s+)?for\s+image(?:[- ]generation)?\s+failover\b",
            r"\b(?:ready|arm|prepare)\s+(?:the\s+)?failover\s+test\b"
        ],
        "answer": "This router can acknowledge the failover test and check whether the primary and fallback image backends are currently reachable. If B-PHONE3 is taken offline, the actual fallback decision will occur on the next image-generation request when backend availability is verified."
    },
    {
    "intent": "model_selection",
    "patterns": [
        r"\bhow\s+do\s+you\s+choose\s+a\s+model\b",
        r"\bhow\s+does\s+the\s+router\s+select\s+a\s+model\b",
        r"\bhow\s+does\s+the\s+router\s+choose\s+which\s+model\b",
        r"\bhow\s+does\s+the\s+router\s+choose\s+an\s+inference\s+node\b",
        r"\bhow\s+do\s+you\s+select\s+the\s+model\b"
    ],
    "answer": "The router selects models and inference nodes using a multi-stage decision process based on intent, context size, and available hardware resources. First, each request is classified through an intent matrix, which determines the appropriate processing lane (such as chat, coding, vision, or tools). A strict precedence ensures that specialized workloads—like vision or code—are routed before general chat. Once the lane is selected, the router chooses the smallest capable model using a ladder strategy. Lightweight nodes (such as phone-based devices like B-PHONE0) are preferred for simple tasks. If the request requires more capability, the workload can escalate to CPU or GPU nodes. The system also maintains session continuity, avoiding unnecessary node switching and preserving context efficiently. For coding tasks, the router behaves like a tool by sending only the relevant user input instead of the full conversation, improving performance and latency. This approach allows the system to balance performance, resource usage, and energy efficiency while maintaining consistent and predictable behavior."
    },
    {
        "intent": "why_not_one_model",
        "patterns": [
            r"\bwhy\s+not\s+use\s+one\s+large\s+model\b",
            r"\bwhy\s+not\s+one\s+model\b",
            r"\bwhy\s+not\s+use\s+a\s+single\s+large\s+model\b",
        ],
        "answer": "A single large model is inefficient for every workload. The LAB Router improves efficiency by matching lightweight tasks to smaller models and escalating only when the workload truly requires stronger compute."
    },
    {
        "intent": "differentiator",
        "patterns": [
            r"\bwhat\s+makes\s+you\s+different\s+from\s+a\s+normal\s+chatbot\b",
            r"\bwhat\s+makes\s+this\s+different\s+from\s+a\s+normal\s+chatbot\b",
            r"\bwhat\s+makes\s+you\s+different\b",
        ],
        "answer": "I do not rely on one model. I orchestrate multiple specialized models and nodes, applying telecom-inspired control-plane logic to AI workloads."
    },
    {
        "intent": "phone0",
        "patterns": [
            r"\bwhat\s+is\s+b-phone0\b",
            r"\bwhat\s+does\s+b-phone0\s+do\b",
        ],
        "answer": "B-PHONE0 is the edge chat node used for lightweight conversational tasks and fast responses."
    },
    {
        "intent": "phone1",
        "patterns": [
            r"\bwhat\s+is\s+b-phone1\b",
            r"\bwhat\s+does\s+b-phone1\s+do\b",
            r"\bwhat\s+is\s+the\s+memory\s+gate\b",
        ],
        "answer": "B-PHONE1 acts as the memory gate. It helps decide whether information should be stored or ignored."
    },
    {
        "intent": "phone2",
        "patterns": [
            r"\bwhat\s+is\s+b-phone2\b",
            r"\bwhat\s+does\s+b-phone2\s+do\b",
        ],
        "answer": "B-PHONE2 is the tools node used for helper tasks and lightweight structured requests."
    },
    {
        "intent": "gpu0",
        "patterns": [
            r"\bwhat\s+is\s+b-gpu0\b",
            r"\bwhat\s+does\s+b-gpu0\s+do\b",
        ],
        "answer": "B-GPU0 is the vision and heavy inference node used for multimodal and compute-intensive workloads."
    },
    {
        "intent": "cpu0",
        "patterns": [
            r"\bwhat\s+is\s+b-cpu0\b",
            r"\bwhat\s+does\s+b-cpu0\s+do\b",
        ],
        "answer": "B-CPU0 is the reasoning and long-context node used for deep processing, large prompts, and coding-oriented tasks."
    },
    {
        "intent": "voice_router",
        "patterns": [
            r"\bhow\s+does\s+voice\s+fit\s+into\s+this\s+architecture\b",
            r"\bwhat\s+is\s+the\s+voice\s+router\b",
            r"\bwhat\s+is\s+the\s+voice\s+fastapi\s+router\b",
        ],
        "answer": "Voice services are handled by a separate Voice FastAPI Router, which forwards text-to-speech requests to the lab voice engine, including Kokoro TTS."
    },
    {
        "intent": "voice_separation",
        "patterns": [
            r"\bdoes\s+voice\s+go\s+through\s+lab\s+router\b",
            r"\bdoes\s+tts\s+go\s+through\s+lab\s+router\b",
            r"\bwhy\s+separate\s+the\s+voice\s+router\b",
        ],
        "answer": "In this architecture, voice synthesis is handled by a separate Voice FastAPI Router so that cognitive routing and speech services remain modular and specialized."
    },
    {
        "intent": "live_demo",
        "patterns": [
            r"\bwhat\s+are\s+we\s+seeing\s+right\s+now\b",
        ],
        "answer": "You are seeing a telecom-inspired cognitive control plane for AI workloads, not just a chatbot."
    },
    {
        "intent": "decision_process",
        "patterns": [
            r"\bhow\s+does\s+the\s+lab\s+router\s+make\s+decisions\b",
            r"\bhow\s+do\s+you\s+make\s+decisions\b",
            r"\bhow\s+do\s+you\s+decide\s+which\s+model\s+to\s+use\b"
        ],
        "answer": "I analyze each request using a control-plane style decision process inspired by telecom systems. I consider the task type, context size, multimodal requirements, routing policy, and node capability before selecting the most appropriate model and compute node. Lightweight conversational tasks stay on edge nodes, while more demanding requests such as coding, vision, or long-context reasoning are escalated to stronger resources only when necessary."
    },
{
    "intent": "memory_optimization",
    "patterns": [
        r"\bhow\s+does\s+the\s+router\s+optimize\s+memory\b",
        r"\bhow\s+does\s+the\s+router\s+manage\s+kv\s*cache\b",
        r"\bhow\s+does\s+the\s+router\s+handle\s+context\s+memory\b"
    ],
    "answer": "The router uses multiple model presets with different context limits. By selecting the smallest suitable preset for a session, it avoids allocating large KV-cache buffers when they are not needed. This reduces memory pressure and helps the system support more concurrent workloads."
    },
    {
    "intent": "energy_efficiency",
    "patterns": [
        r"\bhow\s+does\s+the\s+router\s+reduce\s+compute\b",
        r"\bhow\s+does\s+the\s+router\s+reduce\s+energy\b",
        r"\bhow\s+does\s+the\s+router\s+save\s+energy\b",
        r"\bhow\s+does\s+the\s+router\s+optimize\s+compute\b",
        r"\bwhy\s+does\s+the\s+router\s+use\s+different\s+nodes\b",
        r"\bhow\s+does\s+the\s+router\s+choose\s+inference\s+nodes\b",
        r"\bdoes\s+the\s+router\s+use\s+low\s+power\s+devices\b",
    ],
    "answer": "The router can route requests to inference nodes with different capabilities and resource profiles. Instead of always using the most powerful node, it can start with smaller or lower-power nodes when the workload allows it. For example, in the current cluster configuration the router may start by using a lightweight node such as B-PHONE0 (an older phone device) for simple tasks. If the request requires more capability, the router can route the request to a more powerful node."
    },
    {
        "intent": "context_too_large",
        "patterns": [
            r"\bwhat\s+happens\s+when\s+context\s+becomes\s+too\s+large\b",
            r"\bwhat\s+happens\s+when\s+the\s+context\s+is\s+too\s+large\b",
            r"\bwhat\s+happens\s+when\s+the\s+request\s+exceeds\s+context\b",
            r"\bhow\s+do\s+you\s+handle\s+long\s+context\b"
        ],
        "answer": "When the conversational context becomes large, I first optimize it by rebuilding or trimming the context to keep only the most relevant information. For long inputs, I can also switch to a one-shot processing mode to avoid unnecessary context overhead. If the workload still exceeds the capabilities of the current node, I can prewarm a more capable node and perform a handover so the session continues on stronger hardware. This approach allows me to prioritize efficiency while still supporting long-context workloads when escalation becomes necessary."
    }
]


def _scci_system_knowledge_match(text: str):
    if not SCCI_SYSTEM_KNOWLEDGE_ENABLE:
        return None
    q = (text or "").strip().lower()
    if not q:
        return None
    for item in SCCI_SYSTEM_QA:
        for pattern in item.get("patterns", []):
            try:
                if re.search(pattern, q, flags=re.IGNORECASE):
                    return {"intent": item["intent"], "answer": item["answer"]}
            except Exception:
                continue
    return None

QA_NATURAL_DELAY_ENABLED = int(os.getenv("QA_NATURAL_DELAY_ENABLED", "1"))
QA_NATURAL_DELAY_MIN_MS = int(os.getenv("QA_NATURAL_DELAY_MIN_MS", "250"))
QA_NATURAL_DELAY_MAX_MS = int(os.getenv("QA_NATURAL_DELAY_MAX_MS", "900"))
QA_NATURAL_DELAY_PER_CHAR_MS = float(os.getenv("QA_NATURAL_DELAY_PER_CHAR_MS", "3.0"))
QA_NATURAL_DELAY_JITTER_MS = int(os.getenv("QA_NATURAL_DELAY_JITTER_MS", "120"))

# Optional progressive streaming for deterministic AQ / direct router answers.
# Default OFF to preserve current behavior unless explicitly enabled.
SCCI_DIRECT_STREAM_PROGRESSIVE = int(os.getenv("SCCI_DIRECT_STREAM_PROGRESSIVE", "1"))
SCCI_DIRECT_STREAM_CHUNK_MODE = os.getenv("SCCI_DIRECT_STREAM_CHUNK_MODE", "word").strip().lower()
SCCI_DIRECT_STREAM_WORDS_PER_CHUNK = max(1, int(os.getenv("SCCI_DIRECT_STREAM_WORDS_PER_CHUNK", "2")))
SCCI_DIRECT_STREAM_CHARS_PER_CHUNK = max(8, int(os.getenv("SCCI_DIRECT_STREAM_CHARS_PER_CHUNK", "24")))
SCCI_DIRECT_STREAM_DELAY_MS = max(0, int(os.getenv("SCCI_DIRECT_STREAM_DELAY_MS", "35")))
SCCI_DIRECT_STREAM_PUNCT_DELAY_MS = max(0, int(os.getenv("SCCI_DIRECT_STREAM_PUNCT_DELAY_MS", "90")))
SCCI_DIRECT_STREAM_MAX_CHUNKS = max(1, int(os.getenv("SCCI_DIRECT_STREAM_MAX_CHUNKS", "512")))

async def _qa_natural_delay(text: str = ""):
    """Small human-like delay for regex/direct QA replies only."""
    if not QA_NATURAL_DELAY_ENABLED:
        return
    try:
        n = len((text or "").strip())
        base_ms = int(QA_NATURAL_DELAY_MIN_MS + (n * QA_NATURAL_DELAY_PER_CHAR_MS))
        jitter_ms = random.randint(0, max(0, QA_NATURAL_DELAY_JITTER_MS))
        total_ms = min(max(0, QA_NATURAL_DELAY_MAX_MS), max(0, base_ms + jitter_ms))
        if total_ms > 0:
            await asyncio.sleep(total_ms / 1000.0)
    except Exception:
        return


def _scci_direct_stream_chunks(text: str):
    """Yield small deterministic chunks for AQ/direct responses.

    This preserves factual stability because the full answer is known upfront,
    while optionally presenting it progressively like an LLM stream.
    """
    s = str(text or "")
    if not s:
        return

    mode = SCCI_DIRECT_STREAM_CHUNK_MODE
    emitted = 0

    if mode == "char":
        step = max(8, SCCI_DIRECT_STREAM_CHARS_PER_CHUNK)
        for i in range(0, len(s), step):
            yield s[i:i + step]
            emitted += 1
            if emitted >= SCCI_DIRECT_STREAM_MAX_CHUNKS:
                rest = s[i + step:]
                if rest:
                    yield rest
                return
        return

    words = s.split()
    if not words:
        return
    step = max(1, SCCI_DIRECT_STREAM_WORDS_PER_CHUNK)
    idx = 0
    while idx < len(words):
        chunk_words = words[idx:idx + step]
        chunk = " ".join(chunk_words)
        if idx + step < len(words):
            chunk += " "
        yield chunk
        emitted += 1
        if emitted >= SCCI_DIRECT_STREAM_MAX_CHUNKS:
            if idx + step < len(words):
                rest = " ".join(words[idx + step:])
                if rest:
                    yield rest
            return
        idx += step


async def _scci_direct_stream_delay_for_chunk(chunk: str):
    try:
        delay_ms = SCCI_DIRECT_STREAM_DELAY_MS
        if chunk and chunk.rstrip().endswith((".", "!", "?", ";", ":", ",")):
            delay_ms += SCCI_DIRECT_STREAM_PUNCT_DELAY_MS
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
    except Exception:
        return

def _build_direct_chat_response(req_id: str, content: str):
    now_ts = int(time.time())
    completion_tokens = max(1, int(len(content or "") / 4))
    return {
        "id": f"chatcmpl-scci-{req_id}",
        "object": "chat.completion",
        "created": now_ts,
        "model": "scci-system-knowledge",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": completion_tokens, "total_tokens": completion_tokens},
    }

def _scci_direct_streaming_response(req_id: str, content: str):
    async def _event_iter():
        await _qa_natural_delay(content)
        created = int(time.time())
        chunk1 = {"id": f"chatcmpl-scci-{req_id}", "object": "chat.completion.chunk", "created": created, "model": "scci-system-knowledge", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk1, ensure_ascii=False)}\n\n".encode("utf-8")

        if SCCI_DIRECT_STREAM_PROGRESSIVE:
            for piece in _scci_direct_stream_chunks(content):
                chunk_piece = {"id": f"chatcmpl-scci-{req_id}", "object": "chat.completion.chunk", "created": created, "model": "scci-system-knowledge", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk_piece, ensure_ascii=False)}\n\n".encode("utf-8")
                await _scci_direct_stream_delay_for_chunk(piece)
        else:
            chunk2 = {"id": f"chatcmpl-scci-{req_id}", "object": "chat.completion.chunk", "created": created, "model": "scci-system-knowledge", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk2, ensure_ascii=False)}\n\n".encode("utf-8")

        chunk3 = {"id": f"chatcmpl-scci-{req_id}", "object": "chat.completion.chunk", "created": created, "model": "scci-system-knowledge", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(chunk3, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    return StreamingResponse(_event_iter(), media_type="text/event-stream")

from fastapi.staticfiles import StaticFiles
import base64
import io

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

def _image_gen_output_dir() -> Path:
    p = Path(IMAGE_GEN_OUTPUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _strip_data_url_prefix(s: str) -> str:
    s = str(s or "")
    if "," in s and s.lstrip().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def _image_gen_backend_specs() -> dict:
    return {
        "B-PHONE3-image-engine": {
            "node": IMAGE_GEN_BPHONE3_NODE_NAME,
            "kind": "stable-diffusion.cpp",
            "url": IMAGE_GEN_BPHONE3_URL,
            "endpoint": IMAGE_GEN_BPHONE3_ENDPOINT,
            "steps": int(IMAGE_GEN_BPHONE3_STEPS),
            "sample_method": IMAGE_GEN_BPHONE3_SAMPLE_METHOD,
            "cfg_scale": float(IMAGE_GEN_BPHONE3_CFG_SCALE),
        },
        "B-GPU0-image-engine": {
            "node": IMAGE_GEN_GPU_NODE_NAME,
            "kind": "automatic1111",
            "url": IMAGE_GEN_GPU_URL,
            "endpoint": IMAGE_GEN_GPU_ENDPOINT,
            "steps": int(IMAGE_GEN_GPU_STEPS),
            "sampler_name": IMAGE_GEN_GPU_SAMPLER,
        },
    }


def _image_gen_payload_for_backend(spec: dict, prompt: str) -> dict:
    kind = str(spec.get("kind") or "")
    if kind == "stable-diffusion.cpp":
        return {
            "prompt": prompt,
            "steps": int(spec.get("steps") or 1),
            "width": int(IMAGE_GEN_WIDTH),
            "height": int(IMAGE_GEN_HEIGHT),
            "cfg_scale": float(spec.get("cfg_scale") or 1.0),
            "sample_method": str(spec.get("sample_method") or "euler_a"),
        }
    return {
        "prompt": prompt,
        "steps": int(spec.get("steps") or IMAGE_GEN_STEPS),
        "width": int(IMAGE_GEN_WIDTH),
        "height": int(IMAGE_GEN_HEIGHT),
        "sampler_name": str(spec.get("sampler_name") or IMAGE_GEN_GPU_SAMPLER),
        "batch_size": 1,
        "n_iter": 1,
    }


def _extract_generated_images_from_response(data: dict) -> list[str]:
    if not isinstance(data, dict):
        return []
    images = data.get("images") or []
    out = [x for x in images if isinstance(x, str) and x.strip()]
    if out:
        return out
    for key in ("data", "results"):
        items = data.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for k in ("base64", "image", "b64_json"):
                        v = item.get(k)
                        if isinstance(v, str) and v.strip():
                            out.append(v)
                elif isinstance(item, str) and item.strip():
                    out.append(item)
    for k in ("image", "base64", "b64_json"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


async def _call_image_generation_backend(req_id: str, prompt: str) -> dict:
    backends = _image_gen_backend_specs()
    errors = []
    async with httpx.AsyncClient(timeout=max(60.0, READ_TIMEOUT_S)) as client:
        for backend_name in IMAGE_GEN_BACKEND_ORDER:
            spec = backends.get(backend_name)
            if not spec:
                errors.append(f"{backend_name}:unknown_backend")
                continue
            if not await _image_backend_is_up(spec):
                log.warning(f"[{req_id}] [MEDIA] image_backend_unreachable backend={backend_name}")
                errors.append(f"{backend_name}:unreachable")
                continue
            if not spec:
                errors.append(f"unknown_backend:{backend_name}")
                continue
            payload = _image_gen_payload_for_backend(spec, prompt)
            endpoint = f"{str(spec.get('url') or '').rstrip('/')}{str(spec.get('endpoint') or '')}"
            log.info(f"[{req_id}] SCCI ROUTE → {spec.get('node')}")
            log.info(f"[{req_id}] SCCI IMAGE_GEN_CALL backend={backend_name} kind={spec.get('kind')} endpoint={endpoint} size={IMAGE_GEN_WIDTH}x{IMAGE_GEN_HEIGHT} steps={payload.get('steps')} sampler={payload.get('sampler_name') or payload.get('sample_method')}")
            try:
                r = await client.post(endpoint, json=payload)
                r.raise_for_status()
                data = r.json()
                images = _extract_generated_images_from_response(data)
                if not images:
                    raise RuntimeError(f"{backend_name} returned no images")
                return {"backend": backend_name, "node": str(spec.get("node") or backend_name), "images": images, "raw": data}
            except Exception as e:
                msg = f"{backend_name}:{type(e).__name__}:{e}"
                errors.append(msg)
                log.warning(f"[{req_id}] SCCI IMAGE_GEN_BACKEND_FAIL {msg}")
                continue
    raise RuntimeError("image_generation_all_backends_failed: " + " | ".join(errors))


def _save_generated_image(req_id: str, b64_image: str) -> tuple[str, str]:
    out_dir = _image_gen_output_dir()
    filename = f"img_{req_id}_{uuid.uuid4().hex[:8]}.png"
    file_path = out_dir / filename
    file_path.write_bytes(base64.b64decode(_strip_data_url_prefix(b64_image)))
    public_url = f"{IMAGE_GEN_PUBLIC_BASE}/generated_images/{filename}"
    return str(file_path), public_url


def _build_image_markdown_reply(url: str, backend: str = "") -> str:
    if backend:
        return f"I generated this image on {backend}:\n\n![generated image]({url})"
    return f"I generated this image:\n\n![generated image]({url})"


def _architecture_diagram_path() -> Path:
    return _image_gen_output_dir() / ARCH_DIAGRAM_FILENAME

def _architecture_diagram_public_url() -> str:
    return f"{IMAGE_GEN_PUBLIC_BASE}/generated_images/{ARCH_DIAGRAM_FILENAME}"

def _build_architecture_markdown_reply(url: str) -> str:
    return f"Here is the LAB Router architecture:\n\n![LAB Router Architecture]({url})"


def _vision_preprocess_bg_rgb() -> tuple[int, int, int]:
    if str(ROUTER_VISION_PREPROCESS_BG).lower() in ("white", "#ffffff", "fff"):
        return (255, 255, 255)
    return (0, 0, 0)


def _resize_image_bytes_square_letterbox(img_bytes: bytes, size: int = 896) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow not available")
    target = int(size or 896)
    bg = _vision_preprocess_bg_rgb()
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        src_w, src_h = im.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError("invalid_image_size")
        scale = min(target / float(src_w), target / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        im = im.resize((new_w, new_h), resampling)
        canvas = Image.new("RGB", (target, target), bg)
        off_x = (target - new_w) // 2
        off_y = (target - new_h) // 2
        canvas.paste(im, (off_x, off_y))
        out = io.BytesIO()
        canvas.save(
            out,
            format="JPEG",
            quality=max(50, min(100, int(ROUTER_VISION_PREPROCESS_JPEG_QUALITY or 92))),
            optimize=True,
        )
        return out.getvalue()


def _resize_data_url_image_square_letterbox(data_url: str, size: int = 896) -> str:
    s = str(data_url or "")
    m = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", s, flags=re.DOTALL)
    if not m:
        return s
    raw = base64.b64decode(m.group(2))
    resized = _resize_image_bytes_square_letterbox(raw, size=int(size or 896))
    new_b64 = base64.b64encode(resized).decode("ascii")
    return f"data:image/jpeg;base64,{new_b64}"


def _prepare_multimodal_images_square(messages: list[dict], size: int = 896) -> tuple[list[dict], int, int]:
    if not isinstance(messages, list):
        return list(messages or []), 0, 0
    out: list[dict] = []
    seen = 0
    rewritten = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        m2 = dict(msg)
        content = m2.get("content")
        if not isinstance(content, list):
            out.append(m2)
            continue
        new_content = []
        for part in content:
            if not isinstance(part, dict):
                new_content.append(part)
                continue
            p2 = dict(part)
            ptype = str(p2.get("type") or "").lower()
            try:
                if ptype == "image_url":
                    image_url = p2.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                        if isinstance(url, str) and url.startswith("data:image/"):
                            seen += 1
                            p2["image_url"] = dict(image_url)
                            p2["image_url"]["url"] = _resize_data_url_image_square_letterbox(url, size=size)
                            rewritten += 1
                elif ptype == "input_image":
                    image_url = p2.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                        if isinstance(url, str) and url.startswith("data:image/"):
                            seen += 1
                            p2["image_url"] = dict(image_url)
                            p2["image_url"]["url"] = _resize_data_url_image_square_letterbox(url, size=size)
                            rewritten += 1
                    else:
                        url = p2.get("image_url") or p2.get("url")
                        if isinstance(url, str) and url.startswith("data:image/"):
                            seen += 1
                            new_url = _resize_data_url_image_square_letterbox(url, size=size)
                            if "image_url" in p2:
                                p2["image_url"] = new_url
                            else:
                                p2["url"] = new_url
                            rewritten += 1
            except Exception:
                new_content.append(part)
                continue
            new_content.append(p2)
        m2["content"] = new_content
        out.append(m2)
    return out, seen, rewritten


ROUTER_LOG_MEMORY_WRITES = os.getenv("ROUTER_LOG_MEMORY_WRITES", "0").strip() in ("1", "true", "True", "yes", "YES")

# ---------------- Paths (backend) ----------------
OPENAI_CHAT_PATH = os.getenv("OPENAI_CHAT_PATH", "/v1/chat/completions")
OPENAI_MODELS_PATH = os.getenv("OPENAI_MODELS_PATH", "/v1/models")

HEALTH_PATHS = [p.strip() for p in os.getenv("ROUTER_HEALTH_PATHS", "/health,/v1/models").split(",") if p.strip()]
READY_PATH = os.getenv("ROUTER_READY_PATH", "/ready")
ROUTER_USE_READY = os.getenv("ROUTER_USE_READY", "0").strip() in ("1","true","True","yes","YES")
METRICS_PATH = os.getenv("ROUTER_METRICS_PATH", "/metrics")

# ---------------- Timeouts ----------------
CONNECT_TIMEOUT_S = float(os.getenv("ROUTER_CONNECT_TIMEOUT_S", "2.0"))
READ_TIMEOUT_S = float(os.getenv("ROUTER_READ_TIMEOUT_S", "120.0"))
STREAM_READ_TIMEOUT_S = float(os.getenv("ROUTER_STREAM_READ_TIMEOUT_S", "300.0"))

# ---------------- Core knobs ----------------
DISCOVERY_INTERVAL_S = float(os.getenv("ROUTER_DISCOVERY_INTERVAL_S", "10.0"))
HEALTH_TTL_S = float(os.getenv("ROUTER_HEALTH_TTL_S", "8.0"))
HEALTH_MIN_INTERVAL_S = float(os.getenv("ROUTER_HEALTH_MIN_INTERVAL_S", "3.0"))
FAIL_COOLDOWN_S = float(os.getenv("ROUTER_FAIL_COOLDOWN_S", "12.0"))

METRICS_TTL_S = float(os.getenv("ROUTER_METRICS_TTL_S", "2.0"))
METRICS_MIN_INTERVAL_S = float(os.getenv("ROUTER_METRICS_MIN_INTERVAL_S", "0.8"))

SESSION_TTL_S = float(os.getenv("ROUTER_SESSION_TTL_S", "1800"))

# ---------------- Capability routing (v9) ----------------
# This router keeps a per-session "capability_level" (1/2/3).
# Level 1: lightweight chat (BN10 first)
# Level 2: engineering / code / deeper reasoning (GPU strong model preferred)
# Level 3: long context / complex (prefer bigger context nodes)
#
# KISS knobs (optional env overrides):
CAP_LEVEL_DEFAULT = int(os.getenv("ROUTER_CAP_LEVEL_DEFAULT", "1"))
CAP_DOWNGRADE_TURNS = int(os.getenv("ROUTER_CAP_DOWNGRADE_TURNS", "3"))
CAP_IDLE_TIMEOUT_S = float(os.getenv("ROUTER_CAP_IDLE_TIMEOUT_S", "1800"))
CAP_SIMPLE_MAX_CHARS = int(os.getenv("ROUTER_CAP_SIMPLE_MAX_CHARS", "320"))

# Capacity ladder thresholds (BN10 context pressure)
CAP_CTX_PRESSURE_RATIO = float(os.getenv("ROUTER_CAP_CTX_PRESSURE_RATIO", "0.70"))  # switch BN10 -> PC-CPU same model
CAP_LONGCTX_RATIO = float(os.getenv("ROUTER_CAP_LONGCTX_RATIO", "0.75"))            # escalate to level 3

# Preferred model IDs (should match EXACT /v1/models ids for coherence across nodes)
CHAT_MODEL_ID = os.getenv("ROUTER_CHAT_MODEL_ID", "gemma3-1b-chat@2k-q8_0").strip()
ENGINEERING_MODEL_ID = os.getenv("ROUTER_ENGINEERING_MODEL_ID", "DeepSeek-R1-Distill-Llama-8B-Q5_K_M").strip()
LONGCTX_MODEL_ID = os.getenv("ROUTER_LONGCTX_MODEL_ID", CHAT_MODEL_ID).strip()

# Deep chat / reasoning (CPU-first): prefer this model on PC-CPU for "deep" bucket when GPU is not required.
DEEP_CPU_MODEL_ID = os.getenv("ROUTER_DEEP_CPU_MODEL_ID", "qwen3.5-2b-chat@32k-q5km").strip()

# Deep understanding (GPU): prefer this model on PC-GPU for "deep" bucket when GPU is used/available.
DEEP_GPU_MODEL_ID = os.getenv("ROUTER_DEEP_GPU_MODEL_ID", "Gemma-3-12B-Text").strip()

# Vision: prefer this model for image requests.
VISION_MODEL_ID = os.getenv("ROUTER_VISION_MODEL_ID", "Gemma-3-4B-Vision").strip()

# Optional: allow different preferred models for vision_fast vs vision_reasoning
VISION_FAST_MODEL_ID = os.getenv("ROUTER_VISION_FAST_MODEL_ID", "gemma3-4b-vision@1k-q4km").strip()
# Vision ctx ladder (4B vision only). Defaults work without env; override if your aliases differ.
VISION_4B_1K = os.getenv("ROUTER_VISION_4B_1K", "gemma3-4b-vision@1k-q4km").strip()
VISION_4B_2K = os.getenv("ROUTER_VISION_4B_2K", "gemma3-4b-vision@2k-q4km").strip()
VISION_4B_4K = os.getenv("ROUTER_VISION_4B_4K", "gemma3-4b-vision@4k-q4km").strip()
VISION_4B_8K = os.getenv("ROUTER_VISION_4B_8K", "gemma3-4b-vision@8k-q4km").strip()

ROUTER_VISION_LADDER_ENABLE = int(os.getenv("ROUTER_VISION_LADDER_ENABLE", "1"))
ROUTER_VISION_LADDER_LOG = int(os.getenv("ROUTER_VISION_LADDER_LOG", "1"))


# Gemma3 mmproj budgeting: image -> ~256 tokens (fixed), not base64-length dependent
ROUTER_VISION_TOKENS_PER_IMAGE = int(os.getenv("ROUTER_VISION_TOKENS_PER_IMAGE", "256"))

ROUTER_VISION_TEXT_MULT = float(os.getenv('ROUTER_VISION_TEXT_MULT','1.8'))  # safety inflation for vision text estimate
# Trim vision history to avoid ctx creep across multiple image turns (system msgs preserved)
ROUTER_VISION_HISTORY_TURNS = int(os.getenv("ROUTER_VISION_HISTORY_TURNS", "6"))

def _count_images_in_messages(msgs) -> int:
    n = 0
    if not isinstance(msgs, list):
        return 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    n += 1
    return n

def _trim_messages_for_vision(msgs, keep_turns: int):
    if not isinstance(msgs, list) or keep_turns <= 0:
        return msgs
    if len(msgs) <= keep_turns:
        return msgs
    system = [m for m in msgs if m.get("role") == "system"]
    rest = [m for m in msgs if m.get("role") != "system"]
    return system + rest[-keep_turns:]
def _approx_b64_chars_in_messages(msgs) -> int:
    total = 0
    if not isinstance(msgs, list):
        return 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t in ("image_url", "input_image"):
                    if t == "image_url":
                        iu = p.get("image_url")
                        if isinstance(iu, dict):
                            url = iu.get("url")
                            if isinstance(url, str) and url.startswith("data:"):
                                total += len(url)
                    if t == "input_image":
                        url = p.get("image_url") or p.get("url") or p.get("data")
                        if isinstance(url, str) and url.startswith("data:"):
                            total += len(url)
    return total

def _pick_vision_model_ladder(tokens_needed: int) -> str:
    """Pick smallest vision ctx tier that fits with headroom."""
    try:
        # CTX_HEADROOM_RATIO is the router's global safety headroom (default ~0.88)
        h = float(CTX_HEADROOM_RATIO)
    except Exception:
        h = 0.88
    if tokens_needed <= int(1024 * h):
        return VISION_4B_1K
    if tokens_needed <= int(2048 * h):
        return VISION_4B_2K
    if tokens_needed <= int(4096 * h):
        return VISION_4B_4K
    return VISION_4B_8K

VISION_REASON_MODEL_ID = os.getenv("ROUTER_VISION_REASON_MODEL_ID", "gemma3-12b-vision@1k-q4km").strip()

# Vision fallback policy (stability first)
# - Primary vision node is B-GPU0.
# - B-CPU0 may be used only as a fallback when GPU0 is unavailable or busy.
# - On CPU fallback, restrict to a small @1k vision model (mmproj cost bounded).
VISION_CPU_FALLBACK_MODEL_ID = os.getenv("ROUTER_VISION_CPU_FALLBACK_MODEL_ID", "gemma3-4b-vision@1k-q4km").strip()

GPU_BUSY_QUEUE_DEPTH = int(os.getenv("ROUTER_GPU_BUSY_QUEUE_DEPTH", "2"))
GPU_BUSY_ACTIVE_REQS  = int(os.getenv("ROUTER_GPU_BUSY_ACTIVE_REQS", "2"))
GPU_BUSY_LAT_EMA_MS   = float(os.getenv("ROUTER_GPU_BUSY_LAT_EMA_MS", "4500"))


# Optional: force GPU for "deep" bucket (0/1). Default 0 = CPU-first deep.
ROUTER_DEEP_FORCE_GPU = os.getenv("ROUTER_DEEP_FORCE_GPU", "0").strip() in ("1","true","True","yes","YES")

# Optional: detect explicit reset phrases (comma-separated)
CAP_RESET_PHRASES = [p.strip().lower() for p in (os.getenv("ROUTER_CAP_RESET_PHRASES",
    "reset,new topic,change topic,let's talk about something else,forget previous,ignore previous").split(",")) if p.strip()]

SESSION_HMAC_KEY = os.getenv("ROUTER_SESSION_HMAC_KEY", "change-me-please")

CHARS_PER_TOKEN = float(os.getenv("ROUTER_CHARS_PER_TOKEN", "4.0"))
CTX_HEADROOM_RATIO = float(os.getenv("ROUTER_CTX_HEADROOM_RATIO", "0.88"))
DEFAULT_MAX_TOKENS = int(os.getenv("ROUTER_DEFAULT_MAX_TOKENS", "256"))
MAX_ATTEMPTS = int(os.getenv("ROUTER_MAX_ATTEMPTS", "8"))

VISION_TOKENS_PER_IMAGE = int(os.getenv("ROUTER_VISION_TOKENS_PER_IMAGE", "256"))

# ---------------- Time / Date Grounding ----------------
TZ_NAME = os.getenv("ROUTER_TZ", "").strip()  # e.g. "Europe/Paris" (optional)
DATETIME_TOOL_ENABLE = int(os.getenv("DATETIME_TOOL_ENABLE", str(TIME_TOOL_ENABLE)))
DATETIME_TOOL_VALIDATE_SIMPLE = int(os.getenv("DATETIME_TOOL_VALIDATE_SIMPLE", "1"))
DATETIME_TOOL_DEFAULT_CITY = os.getenv("DATETIME_TOOL_DEFAULT_CITY", "").strip()


def now_dt():
    if TZ_NAME and ZoneInfo:
        import datetime as _dt
        return _dt.datetime.now(ZoneInfo(TZ_NAME))
    import datetime as _dt
    return _dt.datetime.now().astimezone()


def build_time_system_message() -> Dict[str, str]:
    dt = now_dt()
    return {
        "role": "system",
        "content": (
            f"Time now: {dt.isoformat()} ({dt.tzinfo}).\n"
            "Interpret relative dates (today, tomorrow, yesterday, next week, last month) from this time.\n"
            "If the user asks for the current time/date, answer using the Time now value above."
        )
    }


import unicodedata
from functools import lru_cache
from datetime import timedelta

_DATETIME_COMMON_CITY_TZ = {
    "paris": "Europe/Paris",
    "tokyo": "Asia/Tokyo",
    "new york": "America/New_York",
    "new york city": "America/New_York",
    "nyc": "America/New_York",
    "madrid": "Europe/Madrid",
    "london": "Europe/London",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "berlin": "Europe/Berlin",
    "rome": "Europe/Rome",
    "milan": "Europe/Rome",
    "lisbon": "Europe/Lisbon",
    "dublin": "Europe/Dublin",
    "amsterdam": "Europe/Amsterdam",
    "brussels": "Europe/Brussels",
    "zurich": "Europe/Zurich",
    "geneva": "Europe/Zurich",
    "vienna": "Europe/Vienna",
    "prague": "Europe/Prague",
    "warsaw": "Europe/Warsaw",
    "budapest": "Europe/Budapest",
    "athens": "Europe/Athens",
    "istanbul": "Europe/Istanbul",
    "cairo": "Africa/Cairo",
    "casablanca": "Africa/Casablanca",
    "lagos": "Africa/Lagos",
    "nairobi": "Africa/Nairobi",
    "johannesburg": "Africa/Johannesburg",
    "dubai": "Asia/Dubai",
    "riyadh": "Asia/Riyadh",
    "jerusalem": "Asia/Jerusalem",
    "tel aviv": "Asia/Jerusalem",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata",
    "karachi": "Asia/Karachi",
    "dhaka": "Asia/Dhaka",
    "bangkok": "Asia/Bangkok",
    "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "seoul": "Asia/Seoul",
    "taipei": "Asia/Taipei",
    "jakarta": "Asia/Jakarta",
    "manila": "Asia/Manila",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane",
    "perth": "Australia/Perth",
    "auckland": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
    "honolulu": "Pacific/Honolulu",
    "anchorage": "America/Anchorage",
    "vancouver": "America/Vancouver",
    "toronto": "America/Toronto",
    "montreal": "America/Toronto",
    "ottawa": "America/Toronto",
    "chicago": "America/Chicago",
    "denver": "America/Denver",
    "phoenix": "America/Phoenix",
    "mexico city": "America/Mexico_City",
    "bogota": "America/Bogota",
    "lima": "America/Lima",
    "santiago": "America/Santiago",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "sao paulo": "America/Sao_Paulo",
    "rio de janeiro": "America/Sao_Paulo",
}
_DATETIME_COMMON_ALIASES = {
    "uk": "london",
    "england": "london",
    "great britain": "london",
    "spain": "madrid",
    "france": "paris",
    "japan": "tokyo",
    "usa": "new york",
    "us": "new york",
    "united states": "new york",
    "uae": "dubai",
    "south korea": "seoul",
}
for _city in TIME_TOOL_CITIES:
    _key = re.sub(r"\s+", " ", str(_city or "").strip().lower())
    if _key == "new york city":
        _key = "new york"
    if _key in _DATETIME_COMMON_CITY_TZ:
        continue
    _title = _key.title().replace("Of", "of")
    _probe = [
        f"Europe/{_title.replace(' ', '_')}",
        f"America/{_title.replace(' ', '_')}",
        f"Asia/{_title.replace(' ', '_')}",
        f"Africa/{_title.replace(' ', '_')}",
        f"Australia/{_title.replace(' ', '_')}",
        f"Pacific/{_title.replace(' ', '_')}",
    ]
    if ZoneInfo:
        for _tz in _probe:
            try:
                ZoneInfo(_tz)
                _DATETIME_COMMON_CITY_TZ[_key] = _tz
                break
            except Exception:
                pass

_DATETIME_WEEKDAY_PAT = re.compile(r"\b(next|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_DATETIME_RELATIVE_PAT = re.compile(
    r"\b(today|tomorrow|yesterday|day\s+after\s+tomorrow|day\s+before\s+yesterday|"
    r"next\s+week|last\s+week|next\s+month|last\s+month|next\s+year|last\s+year|"
    r"in\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)|"
    r"\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago|"
    r"next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
_DATETIME_CLOCK_PAT = re.compile(
    r"\b(?P<clock>(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:am|pm)?|noon|midnight)\b",
    re.IGNORECASE,
)
_DATETIME_CURRENT_PAT = re.compile(
    r"^\s*(?:what(?:'s|\s+is)?\s+(?:the\s+)?)?"
    r"(?P<kind>local\s+time\s+and\s+date|local\s+date\s+and\s+time|time\s+and\s+date|date\s+and\s+time|local\s+time|local\s+date|time|date|datetime)"
    r"(?:\s+is\s+it)?(?:\s+(?:for|in|at))?\s*(?P<tail>.+?)?\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_DATETIME_CONVERSION_PATTERNS = [
    re.compile(r"^\s*(?:what\s+time\s+(?:is|will\s+it\s+be)|convert)\s+(?P<when>.+?)\s+(?:in|from)\s+(?P<src>.+?)\s+(?:to|in)\s+(?P<dst>.+?)\s*[?.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<when>.+?)\s+(?:in|from)\s+(?P<src>.+?)\s+(?:to|in)\s+(?P<dst>.+?)\s*[?.!]?\s*$", re.IGNORECASE),
]
_WEEKDAY_TO_INT = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _normalize_geo_name(value: str) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("_", " ").replace("/", " ")
    s = re.sub(r"\b(city|time|date|datetime|timezone|zone|local|current|today|tomorrow|yesterday|now|please|the)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-zA-Z0-9+\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return _DATETIME_COMMON_ALIASES.get(s, s)


@lru_cache(maxsize=1)
def _datetime_timezone_index() -> dict:
    index: dict[str, list[str]] = {}
    if not ZoneInfo:
        return index
    try:
        from zoneinfo import available_timezones
        tzs = sorted(available_timezones())
    except Exception:
        tzs = []
    for tz in tzs:
        norm_tz = _normalize_geo_name(tz)
        if norm_tz:
            index.setdefault(norm_tz, []).append(tz)
        parts = [_normalize_geo_name(p) for p in tz.split("/") if p]
        parts = [p for p in parts if p]
        if not parts:
            continue
        index.setdefault(parts[-1], []).append(tz)
        if len(parts) >= 2:
            index.setdefault(" ".join(parts[-2:]), []).append(tz)
    for key, tz in _DATETIME_COMMON_CITY_TZ.items():
        index.setdefault(_normalize_geo_name(key), []).insert(0, tz)
    return index


def _choose_timezone_from_candidates(name_key: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    uniq = []
    for tz in candidates:
        if tz not in uniq:
            uniq.append(tz)
    if len(uniq) == 1:
        return uniq[0]
    pref = [tz for tz in uniq if tz.startswith(("Europe/", "America/", "Asia/", "Africa/", "Australia/", "Pacific/"))]
    if pref:
        uniq = pref
    if name_key:
        exact_tail = [tz for tz in uniq if _normalize_geo_name(tz.split("/")[-1]) == name_key]
        if exact_tail:
            uniq = exact_tail
    return sorted(uniq)[0] if uniq else None


def _resolve_timezone_for_place(place: str) -> tuple[str | None, str | None]:
    raw = str(place or "").strip()
    if not raw:
        if DATETIME_TOOL_DEFAULT_CITY:
            return _resolve_timezone_for_place(DATETIME_TOOL_DEFAULT_CITY)
        return None, None
    if ZoneInfo:
        try:
            ZoneInfo(raw)
            return raw, raw.replace("_", " ")
        except Exception:
            pass
    key = _normalize_geo_name(raw)
    if not key:
        return None, None
    if key in _DATETIME_COMMON_CITY_TZ:
        tz = _DATETIME_COMMON_CITY_TZ[key]
        disp = "New York" if key == "new york" else key.title()
        return tz, disp
    candidates = _datetime_timezone_index().get(key) or []
    chosen = _choose_timezone_from_candidates(key, candidates)
    if not chosen:
        return None, None
    disp = raw.strip(" ,.?!") or chosen.split("/")[-1].replace("_", " ")
    return chosen, disp


def _add_months(dt_obj, months: int):
    month_index = dt_obj.month - 1 + int(months or 0)
    year = dt_obj.year + month_index // 12
    month = month_index % 12 + 1
    import calendar
    day = min(dt_obj.day, calendar.monthrange(year, month)[1])
    return dt_obj.replace(year=year, month=month, day=day)


def _add_years(dt_obj, years: int):
    try:
        return dt_obj.replace(year=dt_obj.year + int(years or 0))
    except Exception:
        import calendar
        year = dt_obj.year + int(years or 0)
        day = min(dt_obj.day, calendar.monthrange(year, dt_obj.month)[1])
        return dt_obj.replace(year=year, day=day)


def _apply_weekday_shift(base_dt, direction: str, weekday_name: str):
    target = _WEEKDAY_TO_INT.get(str(weekday_name or "").lower())
    if target is None:
        return base_dt
    current = base_dt.weekday()
    if str(direction or "").lower() == "last":
        delta = (current - target) % 7
        delta = 7 if delta == 0 else delta
        return base_dt - timedelta(days=delta)
    delta = (target - current) % 7
    delta = 7 if delta == 0 else delta
    return base_dt + timedelta(days=delta)


def _apply_relative_phrase(base_dt, phrase: str):
    p = re.sub(r"\s+", " ", str(phrase or "").strip().lower())
    if not p:
        return base_dt
    if p == "today":
        return base_dt
    if p == "tomorrow":
        return base_dt + timedelta(days=1)
    if p == "yesterday":
        return base_dt - timedelta(days=1)
    if p == "day after tomorrow":
        return base_dt + timedelta(days=2)
    if p == "day before yesterday":
        return base_dt - timedelta(days=2)
    if p == "next week":
        return base_dt + timedelta(weeks=1)
    if p == "last week":
        return base_dt - timedelta(weeks=1)
    if p == "next month":
        return _add_months(base_dt, 1)
    if p == "last month":
        return _add_months(base_dt, -1)
    if p == "next year":
        return _add_years(base_dt, 1)
    if p == "last year":
        return _add_years(base_dt, -1)
    m = re.match(r"^in\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)$", p)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            return base_dt + timedelta(days=n)
        if unit.startswith("week"):
            return base_dt + timedelta(weeks=n)
        if unit.startswith("month"):
            return _add_months(base_dt, n)
        return _add_years(base_dt, n)
    m = re.match(r"^(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago$", p)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            return base_dt - timedelta(days=n)
        if unit.startswith("week"):
            return base_dt - timedelta(weeks=n)
        if unit.startswith("month"):
            return _add_months(base_dt, -n)
        return _add_years(base_dt, -n)
    m = _DATETIME_WEEKDAY_PAT.match(p)
    if m:
        return _apply_weekday_shift(base_dt, m.group(1), m.group(2))
    return base_dt


def _extract_relative_phrase(text: str) -> str:
    if not text:
        return ""
    matches = list(_DATETIME_RELATIVE_PAT.finditer(str(text)))
    if not matches:
        return ""
    matches.sort(key=lambda m: len(m.group(0)), reverse=True)
    return matches[0].group(0)


def _parse_clock_fragment(text: str) -> tuple[int, int, str]:
    t = str(text or "").strip().lower()
    if not t:
        return 0, 0, ""
    if t == "noon":
        return 12, 0, "12:00"
    if t == "midnight":
        return 0, 0, "00:00"
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", t, flags=re.IGNORECASE)
    if not m:
        raise ValueError("unsupported time expression")
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    suffix = str(m.group(3) or "").lower()
    if suffix == "am":
        hour = 0 if hour == 12 else hour
    elif suffix == "pm":
        hour = 12 if hour == 12 else hour + 12
    if hour > 23 or minute > 59:
        raise ValueError("time out of range")
    return hour, minute, f"{hour:02d}:{minute:02d}"


def _strip_noise_tail(text: str) -> str:
    s = re.sub(r"\b(?:right\s+now|now|please)\b", " ", str(text or ""), flags=re.IGNORECASE)
    s = re.sub(r"[?.!]+$", "", s).strip(" ,")
    return re.sub(r"\s+", " ", s).strip()


def _build_datetime_target(base_dt, relative_phrase: str, clock_text: str):
    dt_obj = _apply_relative_phrase(base_dt, relative_phrase)
    if clock_text:
        hh, mm, _ = _parse_clock_fragment(clock_text)
        dt_obj = dt_obj.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return dt_obj


def _detect_datetime_conversion_request(text: str) -> dict | None:
    t = _strip_noise_tail(text)
    if not t:
        return None
    for pat in _DATETIME_CONVERSION_PATTERNS:
        m = pat.match(t)
        if not m:
            continue
        when_text = _strip_noise_tail(m.group("when") or "")
        src_text = _strip_noise_tail(m.group("src") or "")
        dst_text = _strip_noise_tail(m.group("dst") or "")
        if not when_text or not src_text or not dst_text:
            continue
        src_tz, src_disp = _resolve_timezone_for_place(src_text)
        dst_tz, dst_disp = _resolve_timezone_for_place(dst_text)
        if not src_tz or not dst_tz:
            continue
        relative_phrase = _extract_relative_phrase(when_text)
        clock_match = _DATETIME_CLOCK_PAT.search(when_text)
        clock_text = (clock_match.group("clock") if clock_match else "").strip()
        if not clock_text:
            continue
        return {
            "mode": "conversion",
            "kind": "time",
            "source_tz": src_tz,
            "source_city": src_disp,
            "target_tz": dst_tz,
            "target_city": dst_disp,
            "relative_phrase": relative_phrase,
            "clock_text": clock_text,
            "query_text": t,
        }
    return None


def _detect_datetime_current_request(text: str) -> dict | None:
    t = _strip_noise_tail(text)
    if not t:
        return None
    m = _DATETIME_CURRENT_PAT.match(t)
    if not m:
        return None
    kind_raw = str(m.group("kind") or "").strip().lower().replace("local ", "")
    if kind_raw in ("time and date", "date and time", "datetime"):
        kind = "datetime"
    elif kind_raw == "date":
        kind = "date"
    else:
        kind = "time"
    tail = _strip_noise_tail(m.group("tail") or "")
    relative_phrase = _extract_relative_phrase(tail)
    clock_match = _DATETIME_CLOCK_PAT.search(tail)
    clock_text = (clock_match.group("clock") if clock_match else "").strip()
    place_tail = tail
    if relative_phrase:
        place_tail = re.sub(re.escape(relative_phrase), " ", place_tail, flags=re.IGNORECASE)
    if clock_text:
        place_tail = re.sub(re.escape(clock_text), " ", place_tail, flags=re.IGNORECASE)
    place_tail = re.sub(r"\b(?:for|in|at|on)\b", " ", place_tail, flags=re.IGNORECASE)
    place_tail = _strip_noise_tail(place_tail)
    tz_name = None
    city_display = None
    if place_tail:
        tz_name, city_display = _resolve_timezone_for_place(place_tail)
        if not tz_name:
            return None
    elif DATETIME_TOOL_DEFAULT_CITY:
        tz_name, city_display = _resolve_timezone_for_place(DATETIME_TOOL_DEFAULT_CITY)
    return {
        "mode": "current",
        "kind": kind,
        "tz": tz_name,
        "city": city_display or (DATETIME_TOOL_DEFAULT_CITY or "local time"),
        "relative_phrase": relative_phrase,
        "clock_text": clock_text,
        "query_text": t,
    }


def _detect_datetime_tool_request(text: str) -> dict | None:
    if not int(DATETIME_TOOL_ENABLE or 0):
        return None
    t = str(text or "").strip()
    if not t:
        return None
    conv = _detect_datetime_conversion_request(t)
    if conv:
        return conv
    cur = _detect_datetime_current_request(t)
    if cur:
        return cur
    return None


def _normalize_time_tool_city(city: str) -> str:
    c = re.sub(r"\s+", " ", str(city or "").strip().lower())
    if c in ("nyc", "new york city"):
        return "new york"
    return c


def _detect_time_tool_request(text: str) -> dict | None:
    det = _detect_datetime_tool_request(text)
    if not det or det.get("mode") != "current":
        return None
    tz_name = det.get("tz")
    city = det.get("city")
    if not tz_name or not city:
        return None
    city_key = _normalize_time_tool_city(city)
    return {
        "kind": det.get("kind") or "time",
        "city": city,
        "city_key": city_key,
        "tz": tz_name,
        "mode": "current",
        "relative_phrase": det.get("relative_phrase") or "",
        "clock_text": det.get("clock_text") or "",
    }


def _format_datetime_tool_answer(det: dict) -> str:
    import datetime as _dt
    if not ZoneInfo:
        return "Time zone support is unavailable on this router."
    mode = str(det.get("mode") or "current")
    if mode == "conversion":
        src_tz = det.get("source_tz")
        dst_tz = det.get("target_tz")
        if not src_tz or not dst_tz:
            return "I could not resolve one of the requested time zones."
        base = _dt.datetime.now(ZoneInfo(src_tz))
        target_src = _build_datetime_target(base, det.get("relative_phrase") or "", det.get("clock_text") or "")
        target_dst = target_src.astimezone(ZoneInfo(dst_tz))
        rel = str(det.get("relative_phrase") or "").strip()
        rel_txt = f" {rel}" if rel else ""
        return (
            f"{(det.get('clock_text') or '').strip()}{rel_txt} in {det.get('source_city')} is "
            f"{target_dst.strftime('%A, %d %B %Y, %H:%M')} in {det.get('target_city')} ({dst_tz})."
        )
    tz_name = det.get("tz")
    city = det.get("city") or "local time"
    base = _dt.datetime.now(ZoneInfo(tz_name)) if tz_name else now_dt()
    target_dt = _build_datetime_target(base, det.get("relative_phrase") or "", det.get("clock_text") or "")
    rel = str(det.get("relative_phrase") or "").strip()
    clock_text = str(det.get("clock_text") or "").strip()
    label_bits = [x for x in [rel, clock_text] if x]
    label = (" for " + " ".join(label_bits)) if label_bits else ""
    if str(det.get("kind") or "time") == "date":
        return f"The date in {city}{label} is {target_dt.strftime('%A, %d %B %Y')} ({tz_name or str(target_dt.tzinfo)})."
    if str(det.get("kind") or "time") == "datetime":
        return f"In {city}{label}, the date and time is {target_dt.strftime('%A, %d %B %Y, %H:%M')} ({tz_name or str(target_dt.tzinfo)})."
    return f"The time in {city}{label} is {target_dt.strftime('%H:%M')} ({tz_name or str(target_dt.tzinfo)})."


def _format_time_tool_answer(kind: str, city: str, tz_name: str) -> str:
    return _format_datetime_tool_answer({"mode": "current", "kind": kind, "city": city, "tz": tz_name})


def _unwrap_history_query_text(text: str) -> str:
    t = str(text or "")
    if not t:
        return ""
    m = re.search(r"\bQuery:\s*(.+)$", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return str(m.group(1) or "").strip()
    t = t.strip()
    m2 = re.match(r"^query:\s*(.+)$", t, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        return str(m2.group(1) or "").strip()
    return t


def _current_turn_demo_text(messages: list[dict]) -> str:
    msgs = list(messages or [])
    if not msgs:
        return ""
    current = _current_turn_only_messages(msgs)
    if not current:
        return ""
    last = current[-1]
    if str((last or {}).get("role") or "").lower() != "user":
        return ""
    txt = _extract_text_from_content((last or {}).get("content", ""))
    txt = str(txt or "").strip()
    if not txt:
        return ""
    if _detect_openwebui_internal_task(txt):
        return ""
    if _is_history_query_wrapper(txt):
        txt = _unwrap_history_query_text(txt)
    txt = str(txt or "").strip()
    if not txt:
        return ""
    if _detect_openwebui_internal_task(txt):
        return ""
    return txt

def _normalize_time_tool_city(city: str) -> str:
    c = re.sub(r"\s+", " ", str(city or "").strip().lower())
    if c in ("nyc", "new york city"):
        return "new york"
    return c

def _detect_time_tool_request(text: str) -> dict | None:
    if not int(TIME_TOOL_ENABLE or 0):
        return None
    t = str(text or "").strip()
    if not t:
        return None
    m = _TIME_TOOL_PAT.search(t)
    if not m:
        return None
    kind_raw = str(m.group("kind") or m.group("kind2") or "").strip().lower()
    kind_norm = kind_raw.replace("local ", "")
    if kind_norm in ("time and date", "date and time"):
        kind = "datetime"
    elif kind_norm == "date":
        kind = "date"
    else:
        kind = "time"
    city_raw = str(m.group("city") or "").strip()
    city_key = _normalize_time_tool_city(city_raw)
    tz_name = _TIME_TOOL_CITY_TZ.get(city_key)
    if not tz_name:
        return None
    city_display = "New York" if city_key == "new york" else city_key.title()
    return {"kind": kind, "city": city_display, "city_key": city_key, "tz": tz_name}

def _format_time_tool_answer(kind: str, city: str, tz_name: str) -> str:
    import datetime as _dt
    if not ZoneInfo:
        return "Time zone support is unavailable on this router."
    now = _dt.datetime.now(ZoneInfo(tz_name))
    if kind == "date":
        return f"Today's date in {city} is {now.strftime('%A, %d %B %Y')}."
    if kind == "datetime":
        return f"In {city}, the current date and time is {now.strftime('%A, %d %B %Y, %H:%M')} ({tz_name})."
    return f"The current time in {city} is {now.strftime('%H:%M')} ({tz_name})."

def _extract_json_object_from_text(s: str) -> dict | None:
    txt = str(s or "").strip()
    if not txt:
        return None
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

async def _time_tool_functiongemma_validate(req_id: str, det: dict) -> tuple[bool, str]:
    city = det.get("city") or ""
    kind = det.get("kind") or "time"
    prompt = (
        "You are a strict function-selection model.\n"
        "You must never answer conversationally.\n"
        "You must never apologize.\n"
        "Available tools:\n"
        "1) get_time(city: string)\n"
        "2) get_date(city: string)\n"
        "3) get_datetime(city: string)\n"
        "Return ONLY one raw JSON object.\n"
        "Valid JSON examples:\n"
        '{"tool":"get_time","city":"Paris"}\n'
        '{"tool":"get_date","city":"Tokyo"}\n'
        '{"tool":"get_datetime","city":"New York"}\n'
        "No markdown. No prose. No explanation.\n"
        f"User request: What {('time and date' if kind == 'datetime' else kind)} is it in {city}?"
    )
    chosen_node = None
    for _nn in [FUNCTION_TOOL_NODE_PRIMARY, FUNCTION_TOOL_NODE_FALLBACK]:
        _n = _nodes.get(_nn)
        if not _n:
            continue
        try:
            await check_node_health(_n)
            await discover_models_for_node(_n)
        except Exception:
            pass
        if getattr(_n, "models", None) is None or FUNCTION_TOOL_MODEL_ID in getattr(_n, "models", {}):
            chosen_node = _n
            break
    if not chosen_node:
        return False, "no_function_node"
    payload = {
        "model": FUNCTION_TOOL_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0,
        "max_tokens": 64,
    }
    try:
        log.info(f"[{req_id}] → SCCI TIME_TOOL_VALIDATE sid={req_id} node={chosen_node.name} model={FUNCTION_TOOL_MODEL_ID} city={city} kind={kind}")
        r = await client.post(chosen_node.url(OPENAI_CHAT_PATH), json=payload)
        if r.status_code >= 400:
            log.warning(f"[{req_id}] SCCI TIME_TOOL_VALIDATE_FAIL sid={req_id} node={chosen_node.name} status={r.status_code}")
            return False, f"http_{r.status_code}"
        data = r.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        obj = _extract_json_object_from_text(content) or {}
        tool = str(obj.get("tool") or "").strip().lower()
        city_out = _normalize_time_tool_city(obj.get("city") or city)
        want_tool = "get_datetime" if kind == "datetime" else ("get_date" if kind == "date" else "get_time")
        ok = (tool == want_tool and city_out == det.get("city_key"))
        if ok:
            log.info(f"[{req_id}] → SCCI TIME_TOOL_OK sid={req_id} node={chosen_node.name} tool={tool} city={city_out}")
            return True, chosen_node.name
        log.warning(f"[{req_id}] SCCI TIME_TOOL_MISMATCH sid={req_id} node={chosen_node.name} content={content!r}")
        return False, chosen_node.name
    except Exception as e:
        log.warning(f"[{req_id}] SCCI TIME_TOOL_VALIDATE_ERR sid={req_id} err={type(e).__name__}:{e}")
        return False, type(e).__name__


_SCCI_OPS_PAT = re.compile(
    r"^\s*(?:scci\s+)?(?P<kind>router\s+status|cluster\s+health|node\s+uptime|current\s+time)\s*(?:for\s+(?P<node>[A-Za-z0-9\-]+))?\s*[?.!]?\s*$",
    re.IGNORECASE,
)

def _detect_scci_ops_request(text: str) -> dict | None:
    t = str(text or "").strip()
    if not t:
        return None
    m = _SCCI_OPS_PAT.search(t)
    if not m:
        return None
    kind = str(m.group("kind") or "").strip().lower().replace(" ", "_")
    node = str(m.group("node") or "").strip()
    return {"kind": kind, "node": node}


def _normalize_demo_control_text(text: str) -> str:
    t = str(text or "")
    if not t:
        return ""
    # Normalize apostrophes and punctuation variants commonly produced by keyboards/UIs.
    t = (t.replace("’", "'")
           .replace("‘", "'")
           .replace("`", "'")
           .replace("“", '"')
           .replace("”", '"'))
    t = re.sub(r"[!?]+", " ", t)
    # Normalize B-PHONE3 aliases such as B-Phone-3 / B Phone 3 / bphone3.
    t = re.sub(r"\bb\s*[-_ ]*phone\s*[-_ ]*3\b", "b-phone3", t, flags=re.IGNORECASE)
    # Normalize common wording variants for the demo control phrase.
    t = re.sub(r"\b(?:i am|i'm|im)\b", "i am", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

_DEMO_FAILOVER_CONTROL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:now\s+)?i\s+am\s+going\s+to\s+put\s+(?:my\s+)?b-phone3\s+on\s+airplane\s+mode\b",
        r"\bput\s+(?:my\s+)?b-phone3\s+on\s+airplane\s+mode\b",
        r"\bb-phone3\s+on\s+airplane\s+mode\b",
        r"\b(?:now\s+)?i\s+am\s+going\s+to\s+(?:disconnect|take\s+offline|switch\s+off|turn\s+off)\s+b-phone3\b",
        r"\bsimulate\s+(?:node\s+)?(?:failure|unavailability|offline)\b",
        r"\bprepare\s+(?:the\s+router\s+)?for\s+image(?:[- ]generation)?\s+failover\b",
        r"\bprepare\s+for\s+image(?:[- ]generation)?\s+failover\b",
        r"\b(?:ready|arm|prepare)\s+(?:the\s+)?failover\s+test\b",
    ]
]

def _detect_demo_failover_control_intent(text: str) -> bool:
    t = _normalize_demo_control_text(text)
    if not t:
        return False
    return any(p.search(t) for p in _DEMO_FAILOVER_CONTROL_PATTERNS)

async def _check_image_failover_readiness() -> dict:
    backends = _image_gen_backend_specs()
    primary_name = IMAGE_GEN_BACKEND_ORDER[0] if IMAGE_GEN_BACKEND_ORDER else "B-PHONE3-image-engine"
    fallback_name = IMAGE_GEN_BACKEND_ORDER[1] if len(IMAGE_GEN_BACKEND_ORDER) > 1 else "B-GPU0-image-engine"
    primary_spec = backends.get(primary_name, {})
    fallback_spec = backends.get(fallback_name, {})
    primary_up = await _image_backend_is_up(primary_spec) if primary_spec else False
    fallback_up = await _image_backend_is_up(fallback_spec) if fallback_spec else False
    return {
        "primary_name": primary_name,
        "primary_node": str(primary_spec.get("node") or primary_name),
        "primary_up": bool(primary_up),
        "fallback_name": fallback_name,
        "fallback_node": str(fallback_spec.get("node") or fallback_name),
        "fallback_up": bool(fallback_up),
    }


def _build_demo_failover_control_answer(status: dict) -> str:
    primary_node = str((status or {}).get("primary_node") or "B-PHONE3-image-engine")
    fallback_node = str((status or {}).get("fallback_node") or "B-GPU0-image-engine")
    primary_up = bool((status or {}).get("primary_up"))
    fallback_up = bool((status or {}).get("fallback_up"))
    if primary_up and fallback_up:
        return (
            f"{primary_node} image backend is currently reachable.\n"
            f"{fallback_node} fallback backend is also reachable.\n\n"
            f"You can proceed with the failover test. If {primary_node} becomes unreachable, "
            f"the next image-generation request will be rerouted automatically to {fallback_node}."
        )
    if (not primary_up) and fallback_up:
        return (
            f"{primary_node} image backend is already unreachable.\n"
            f"{fallback_node} fallback backend is reachable.\n\n"
            f"The next image-generation request will use {fallback_node}. You can proceed with the test."
        )
    if primary_up and (not fallback_up):
        return (
            f"{primary_node} image backend is currently reachable, but {fallback_node} fallback is not reachable.\n\n"
            f"Failover may not succeed until the fallback backend is available."
        )
    return (
        f"Neither {primary_node} nor {fallback_node} is currently reachable for image generation.\n\n"
        f"Please verify backend availability before running the failover test."
    )

def _human_uptime(seconds: float) -> str:
    s = max(0, int(seconds or 0))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or parts:
        parts.append(f"{h}h")
    if m or parts:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

def _scci_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat()

def _build_scci_ops_answer(det: dict) -> str:
    kind = str((det or {}).get("kind") or "")
    node_name = str((det or {}).get("node") or "").strip()
    if kind == "current_time":
        return f"SCCI current time is {_scci_now_iso()}."
    if kind == "router_status":
        up = sum(1 for n in _nodes.values() if str(getattr(n, "status", "")) == "ok")
        total = len(_nodes)
        return (
            f"SCCI router status is operational. "
            f"Router uptime: {_human_uptime(time.time() - ROUTER_STARTED_AT)}. "
            f"Known nodes: {total}. Nodes currently up: {up}. "
            f"Discovery interval: {DISCOVERY_INTERVAL_S}s."
        )
    if kind == "cluster_health":
        parts = []
        for n in _nodes.values():
            up = 1 if str(getattr(n, "status", "")) == "ok" else 0
            q = int(getattr(getattr(n, "metrics", None), "queue_depth", 0) or 0)
            act = int(getattr(getattr(n, "metrics", None), "active_requests", 0) or 0)
            parts.append(f"{n.name}=up:{up},ready:{1 if getattr(n, 'ready', False) else 0},q:{q},act:{act},status:{n.status}")
        return "SCCI cluster health: " + " | ".join(parts)
    if kind == "node_uptime":
        if not node_name:
            return "Please specify a node name, for example: scci node uptime for B-GPU0."
        node = _nodes.get(node_name)
        if not node:
            return f"Unknown node: {node_name}."
        first_up = _NODE_FIRST_UP_TS.get(node_name)
        if first_up:
            return f"SCCI node uptime for {node_name}: observed up for {_human_uptime(time.time() - first_up)}."
        return f"SCCI node uptime for {node_name}: no observed uptime yet in this router process."
    return "Unsupported SCCI operations request."

# ---------------- Time Injection Control ----------------
# Time grounding is injected ONLY when user intent needs temporal reasoning.
# We intentionally match broad "past/present/future" language, but we still suppress
# injection for OpenWebUI meta-tasks (title/tags/followups).
_TIME_INTENT_PAT = re.compile(
    r"("
    r"\b(?:time|date|today|tonight|now|right\s+now|currently|asap|immediately)\b|"
    r"\b(?:tomorrow|yesterday|tonight|last\s+night|this\s+(?:morning|afternoon|evening)|earlier\s+today)\b|"
    r"\b(?:next|last)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\b(?:in\s+(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+|couple|few|several)\s*(?:of\s*)?"
    r"(?:min|mins|minute|minutes|h|hr|hour|hours|day|days|week|weeks|month|months|year|years))\b|"
    r"\b(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?:min|mins|minute|minutes|h|hr|hour|hours|day|days|week|weeks|month|months|year|years))\b|"
    r"\b(?:ago|later|soon|eventually)\b|"
    r"\b(?:remind\s+me|reminder|schedule|meeting|deadline|appointment|calendar|eta|due\s+date)\b|"
    r"\b(?:by|before|after|until)\s+\w+\b|"
    r"(?:\bat\s+)?\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
    r")",
    re.IGNORECASE,
)

_OPENWEBUI_TASK_PAT = re.compile(
    r"^\s*#{2,}\s*task\b|"
    r"\bgenerate\s+1-3\s+broad\s+tags\b|"
    r"\bgenerate\s+a\s+concise\b.*\btitle\b|"
    r"\bsuggest\s+3-5\s+relevant\s+follow-?up\s+questions\b|"
    r"\bchat\s*title\b|\bfollow-?up\b\s*questions\b|\btags\b\s*categor",
    re.IGNORECASE | re.MULTILINE,
)

# Time grounding injection policy:
# Default (v9.1.1): inject ONLY when the user explicitly asks for current time/date.
# You can revert to the broader v9 behavior by setting ROUTER_TIME_INJECT_EXPLICIT_ONLY=0.
_TIME_EXPLICIT_PAT = re.compile(
    r"\b(?:what\s+time\s+is\s+it|current\s+time|time\s+now|what\s+date\s+is\s+it|current\s+date|today'?s\s+date|date\s+today)\b",
    re.IGNORECASE,
)
ROUTER_TIME_INJECT_EXPLICIT_ONLY = os.getenv("ROUTER_TIME_INJECT_EXPLICIT_ONLY", "1").strip() not in ("0","false","False","no","NO")

def should_inject_time(user_text: str) -> bool:
    if not user_text:
        return False
    if ROUTER_TIME_INJECT_EXPLICIT_ONLY:
        return bool(_TIME_EXPLICIT_PAT.search(user_text))
    return bool(_TIME_INTENT_PAT.search(user_text))

def is_openwebui_task_prompt(user_text: str) -> bool:
    if not user_text:
        return False
    return bool(_OPENWEBUI_TASK_PAT.search(user_text.strip()))

# ---------------- SQLite Memory ----------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ROUTER_MEMORY_DB", str(BASE_DIR / "brain_storage.db")))

SCOPE_PROFILE = 1
SCOPE_PROJECT = 2
# ---------------------------------------------------------------------------
# v8.1.4+ Hybrid Scope Resolver helper
# ---------------------------------------------------------------------------
# We keep this intentionally simple (KISS):
# - If we have a non-empty chat_id, we treat the session as an "active project"
#   and store SAVE FACT / DIRECTIVE / CORRECTION into PROJECT scope.
# - If chat_id is missing, we store into PROFILE scope.
#
# Later improvement (optional): detect project-ness via explicit metadata fields
# (e.g., OpenWebUI conversation id, X-Conversation-Id header, etc.).
#
def session_has_active_project(chat_id: str | None) -> bool:
    return bool(chat_id)


MEMORY_MAX_PROFILE = int(os.getenv("ROUTER_MEMORY_MAX_PROFILE", "8"))
MEMORY_MAX_PROJECT = int(os.getenv("ROUTER_MEMORY_MAX_PROJECT", "12"))
MEMORY_MAX_CHARS_TOTAL = int(os.getenv("ROUTER_MEMORY_MAX_CHARS_TOTAL", "2400"))

# v8 enable switch
MEMORY_ENABLE_V8 = os.getenv("ROUTER_MEMORY_ENABLE_V8", "1").strip() not in ("0", "false", "False", "no", "NO")

# v8 decay + reinforcement
MEMORY_DECAY_HALFLIFE_DAYS = float(os.getenv("ROUTER_MEMORY_DECAY_HALFLIFE_DAYS", "30"))
MEMORY_RECALL_BOOST = int(os.getenv("ROUTER_MEMORY_RECALL_BOOST", "6"))  # importance boost when recalled
MEMORY_WRITE_DEFAULT_IMPORTANCE = int(os.getenv("ROUTER_MEMORY_WRITE_DEFAULT_IMPORTANCE", "45"))
MEMORY_CANDIDATE_LIMIT = int(os.getenv("ROUTER_MEMORY_CANDIDATE_LIMIT", "250"))

# Gatekeeper (same as v7.5.1)
GATEKEEPER_MODEL_ID = os.getenv("ROUTER_GATEKEEPER_MODEL", "google_gemma-3-1b-it-Q8_0")
GATEKEEPER_MAX_TOKENS = int(os.getenv("ROUTER_GATEKEEPER_MAX_TOKENS", "64"))

# Gatekeeper concurrency control (perf, no behavior change)
# - Prevents mem_gate from becoming a bottleneck or stampeding a small CPU node.
# - Keep this at 1 for single-user / low load (default).
# - Raise to 2-4 only if your BN8/PC-CPU can handle concurrent classifications.
GATEKEEPER_CONCURRENCY = int(os.getenv("ROUTER_GATEKEEPER_CONCURRENCY", "1"))
GATEKEEPER_SEMAPHORE = asyncio.Semaphore(max(1, GATEKEEPER_CONCURRENCY))


# ---------------- Async memory write plumbing (v8.1.6 perf, KISS) ----------------
# Defaults are defined in-code so missing env vars never crash the router.
# Env knobs (all optional):
#   ROUTER_DB_WRITE_ASYNC=1        (0 disables queue+worker; writes happen in-thread via to_thread)
#   ROUTER_DB_WRITE_QUEUE_MAX=2048 (max pending write jobs; if full, we drop writes)
#   ROUTER_DB_WRITE_WORKERS=1      (SQLite writers; keep 1 to avoid lock contention)
#   ROUTER_LOG_MEMORY_WRITES=1     (existing flag; keep behavior)
ROUTER_DB_WRITE_ASYNC = int(os.getenv("ROUTER_DB_WRITE_ASYNC", "1")) != 0
ROUTER_DB_WRITE_QUEUE_MAX = int(os.getenv("ROUTER_DB_WRITE_QUEUE_MAX", "2048"))
ROUTER_DB_WRITE_WORKERS = int(os.getenv("ROUTER_DB_WRITE_WORKERS", "1"))

# Internal queue/tasks (initialized on startup). Always defined to avoid NameError.
_MEMORY_WRITE_QUEUE: Optional[asyncio.Queue] = None
_MEMORY_WRITE_TASKS: List[asyncio.Task] = []


def _enqueue_memory_write(
    scope: int,
    content: str,
    session_id: str,
    user_id: Optional[str],
    chat_id: Optional[str],
    kind: str,
    importance: int,
    confidence: int,
) -> None:
    """Best-effort enqueue of a memory write job.

    KISS rules:
    - Never block the request path.
    - If queue is not ready or full, drop the write (memory is helpful, not critical).
    """
    q = _MEMORY_WRITE_QUEUE
    if q is None:
        return
    job = (scope, content, session_id, user_id, chat_id, kind, importance, confidence)
    try:
        q.put_nowait(job)
    except Exception:
        # QueueFull or event loop issues; drop silently or log if desired.
        if ROUTER_LOG_MEMORY_WRITES:
            log.info("[memory_write] drop (queue full or unavailable)")

async def _memory_write_worker(worker_id: int) -> None:
    """Background worker that flushes queued memory writes to SQLite.

    Uses asyncio.to_thread to avoid blocking the FastAPI event loop on disk I/O.
    """
    q = _MEMORY_WRITE_QUEUE
    if q is None:
        return
    while True:
        job = await q.get()
        try:
            if job is None:
                return
            scope, content, session_id, user_id, chat_id, kind, importance, confidence = job
            await asyncio.to_thread(
                db_insert_memory_v8,
                scope, content, session_id, user_id, chat_id,
                kind, importance, confidence
            )
        except Exception as e:
            log.warning(f"[memory_write_worker:{worker_id}] error: {e!r}")
        finally:
            try:
                q.task_done()
            except Exception:
                pass



# ---------------- Gatekeeper policy (KISS now, scalable later) ----------------
# WHY THIS EXISTS:
# - Today we classify into 3 states (0/1/2) and only save memories for 1/2.
# - Later you may want 5+ states (e.g. add TASK, PREFERENCE, CONSTRAINT, etc.)
#   WITHOUT changing router logic everywhere.
#
# HOW TO UPGRADE LATER (NO CODE CHANGES, ONLY CONFIG):
# 1) Keep using ROUTER_GATEKEEPER_MODEL to swap models (already supported).
# 2) Define additional states via one of:
#    - ROUTER_GATEKEEPER_POLICY_JSON  (inline JSON string)
#    - ROUTER_GATEKEEPER_POLICY_PATH  (path to a JSON file)
# 3) Update your gatekeeper_prompt() text to describe the new states.
# 4) The router will automatically:
#    - update the JSON Schema enum sent to the gatekeeper (strict output)
#    - accept + parse the new states
#    - apply per-state actions for memory saving
#
# IMPORTANT:
# - If no policy override is provided, router uses DEFAULT_GATEKEEPER_POLICY below.
# - Policy parsing is "best effort": invalid JSON will fall back to defaults and NEVER crash the router.


DEFAULT_GATEKEEPER_POLICY = {
    "version": 1,
    # v8.1.4 default 4-state policy:
    # 0 IGNORE     -> do nothing
    # 1 SAVE FACT  -> save to PROFILE or PROJECT (hybrid rule, decided in memory_write_task)
    # 2 DIRECTIVE  -> save directive (hybrid rule)
    # 3 CORRECTION -> save correction marker (hybrid rule)
    #
    # NOTE: We intentionally do NOT force the 1B model to guess "scope".
    # Scope is resolved by router logic:
    #   if session_has_active_project(chat_id): PROJECT else PROFILE
    "states": {
        0: {"name": "ignore",     "action": "ignore"},
        1: {"name": "knowledge",  "action": "save", "kind": "fact"},
        2: {"name": "directive",  "action": "save", "kind": "directive", "importance": 70},
        3: {"name": "correction", "action": "save", "kind": "correction", "importance": 65},
    },
}

_GATEKEEPER_POLICY_CACHE: Optional[Dict[str, Any]] = None

_GATEKEEPER_POLICY_CACHE: Optional[Dict[str, Any]] = None

def load_gatekeeper_policy() -> Dict[str, Any]:
    """Load gatekeeper policy from env/file if provided, else use DEFAULT_GATEKEEPER_POLICY.

    Env overrides:
      - ROUTER_GATEKEEPER_POLICY_JSON : JSON string with {"states": {...}}
      - ROUTER_GATEKEEPER_POLICY_PATH : file path to JSON with {"states": {...}}

    Keys under "states" can be strings or ints; we normalize to int.
    """
    global _GATEKEEPER_POLICY_CACHE
    if _GATEKEEPER_POLICY_CACHE is not None:
        return _GATEKEEPER_POLICY_CACHE

    pol: Dict[str, Any] = dict(DEFAULT_GATEKEEPER_POLICY)

    raw = (os.getenv("ROUTER_GATEKEEPER_POLICY_JSON", "") or "").strip()
    path = (os.getenv("ROUTER_GATEKEEPER_POLICY_PATH", "") or "").strip()

    try:
        if path:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()

        if raw:
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("states"), dict):
                st_norm: Dict[int, Dict[str, Any]] = {}
                for k, v in obj["states"].items():
                    try:
                        ki = int(k)
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        st_norm[ki] = v
                if st_norm:
                    pol.update(obj)
                    pol["states"] = st_norm
    except Exception as e:
        log.warning(f"[gatekeeper_policy] invalid override; using defaults ({e!r})")

    _GATEKEEPER_POLICY_CACHE = pol
    return pol

def gatekeeper_allowed_states() -> List[int]:
    """Return sorted list of allowed integer states for strict schema + parsing."""
    pol = load_gatekeeper_policy()
    st = pol.get("states") or {}
    try:
        allowed = sorted({int(k) for k in st.keys()})
    except Exception:
        allowed = [0, 1, 2]
    return allowed if allowed else [0, 1, 2]

def gatekeeper_rule_for_state(state: Optional[int]) -> Optional[Dict[str, Any]]:
    """Return policy rule dict for a given state, or None if unknown.

    Robust to policy JSON where state keys might be strings ("1") instead of ints (1).
    """
    if state is None:
        return None
    pol = load_gatekeeper_policy()
    st = pol.get("states") or {}
    try:
        key_int = int(state)
    except Exception:
        return None
    if isinstance(st, dict):
        if key_int in st:
            return st.get(key_int)
        sk = str(key_int)
        if sk in st:
            return st.get(sk)
    return None


# -------- Memory poison protection --------
MEMORY_SAVE_MAX_CHARS = int(os.getenv("ROUTER_MEMORY_SAVE_MAX_CHARS", "400"))
MEMORY_SAVE_MAX_NEWLINES = int(os.getenv("ROUTER_MEMORY_SAVE_MAX_NEWLINES", "6"))
MEMORY_SAVE_BLOCK_PATTERNS = [
    r"^\s*History:\b",
    r"^\s*Query:\b",
    r"^\s*#{2,}\s*task\b",
    r"\bchat\s*history\b",
    r"\bassistant:\b",
    r"\bsystem:\b",
    r"\btools?\b.*\bschema\b",
    r"\bjson_schema\b",
    r"\bresponse_format\b",
]
_BLOCK_RE = re.compile("|".join(f"({p})" for p in MEMORY_SAVE_BLOCK_PATTERNS), re.IGNORECASE | re.MULTILINE)

def sanitize_for_memory_save(user_text: str) -> Optional[str]:
    if not user_text:
        return None
    t = user_text.strip()
    t = re.sub(r"[ \t]+", " ", t).strip()
    if not t:
        return None
    if len(t) > MEMORY_SAVE_MAX_CHARS:
        return None
    if t.count("\n") > MEMORY_SAVE_MAX_NEWLINES:
        return None
    if _BLOCK_RE.search(t):
        return None
    if t.startswith("{") and ("schema" in t.lower() or "response_format" in t.lower()):
        return None
    return t

# -------- Memory save eligibility (v9.1.1) --------
# We only want to store durable USER/PROJECT facts and directives, not trivia or assistant answers.
# This is a conservative filter; the LLM gatekeeper still makes the final decision.
_MEM_EXPLICIT_SAVE_PAT = re.compile(r"\b(?:remember\s+this|save\s+this|store\s+this|note\s+this)\b", re.IGNORECASE)
_MEM_PROFILE_FACT_PAT = re.compile(r"\b(?:my\s+name\s+is|i\s+am|i\s+live\s+in|my\s+email|my\s+ip|my\s+address|my\s+phone|my\s+favorite|i\s+prefer|i\s+like|i\s+don't\s+like)\b", re.IGNORECASE)
_MEM_PROJECT_FACT_PAT = re.compile(r"\b(?:we\s+use|we\s+need|we\s+will|our\s+project|project\s+requirement|constraint|router|openwebui|llama\.cpp|node|bn10|pc-cpu|pc-gpu|ctx|context|ip\s*address|port\s*\d+|gpu|rtx|model\s+id)\b", re.IGNORECASE)

def is_memory_save_candidate(user_text: str) -> bool:
    """Cheap prefilter to avoid polluting memory with generic Q/A facts.

    Returns True if the text looks like a user/profile/project fact OR the user explicitly asks to save it.
    """
    t = (user_text or "").strip()
    if not t:
        return False
    # Never store questions (trivia Q/A, help requests, etc.)
    if t.endswith("?"):
        return False
    if _MEM_EXPLICIT_SAVE_PAT.search(t):
        return True
    if _MEM_PROFILE_FACT_PAT.search(t):
        return True
    if _MEM_PROJECT_FACT_PAT.search(t):
        return True
    return False


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def db_init() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Old v7 table (keep for backward compatibility / migration)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope INTEGER NOT NULL,
            user_id TEXT,
            chat_id TEXT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence INTEGER DEFAULT 100,
            created_at INTEGER NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_profile ON memories(scope, user_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_project ON memories(scope, chat_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id, created_at DESC)")

    # v8 table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_v8 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'fact',
            user_id TEXT,
            chat_id TEXT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT,
            importance INTEGER NOT NULL DEFAULT 45,     -- 0..100
            confidence INTEGER NOT NULL DEFAULT 100,    -- 0..100
            access_count INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            last_accessed INTEGER NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_v8_profile ON memories_v8(scope, user_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_v8_project ON memories_v8(scope, chat_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_v8_session ON memories_v8(session_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_v8_hash ON memories_v8(content_hash)")

    con.commit()

    # Auto-migration: if v8 empty and old has rows, copy them
    try:
        cur.execute("SELECT COUNT(*) FROM memories_v8")
        v8_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM memories")
        old_count = int(cur.fetchone()[0])
        if v8_count == 0 and old_count > 0:
            log.info(f"[v8] migrating old memories -> memories_v8 (rows={old_count})")
            # content_hash = sha1(scope|user_id|chat_id|content) to reduce duplicates
            cur.execute("SELECT scope, user_id, chat_id, session_id, content, confidence, created_at FROM memories ORDER BY id ASC")
            rows = cur.fetchall()
            for (scope, user_id, chat_id, session_id, content, confidence, created_at) in rows:
                ch = hashlib.sha1(f"{scope}|{user_id}|{chat_id}|{content}".encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO memories_v8(scope, kind, user_id, chat_id, session_id, content, content_hash,
                                            importance, confidence, access_count, created_at, last_accessed)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        int(scope), "fact",
                        user_id, chat_id, session_id,
                        content, ch,
                        int(MEMORY_WRITE_DEFAULT_IMPORTANCE),
                        int(confidence) if confidence is not None else 100,
                        0,
                        int(created_at),
                        int(created_at),
                    )
                )
            con.commit()
    except Exception as e:
        log.warning(f"[v8] migration skipped due to error: {e!r}")

    con.close()

def _hash_content(scope: int, user_id: Optional[str], chat_id: Optional[str], content: str) -> str:
    return hashlib.sha1(f"{scope}|{user_id}|{chat_id}|{content}".encode("utf-8")).hexdigest()

def db_insert_memory_v8(
    scope: int,
    content: str,
    session_id: str,
    user_id: Optional[str],
    chat_id: Optional[str],
    kind: str = "fact",
    importance: int = 45,
    confidence: int = 100,
) -> None:
    now = int(time.time())
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    ch = _hash_content(scope, user_id, chat_id, content)

    # De-dup: if same hash exists, just reinforce it instead of inserting new
    cur.execute(
        "SELECT id, importance, access_count FROM memories_v8 WHERE content_hash=? ORDER BY id DESC LIMIT 1",
        (ch,)
    )
    row = cur.fetchone()
    if row:
        mid, imp, acc = int(row[0]), int(row[1]), int(row[2])
        new_imp = min(100, max(0, imp + 2))
        cur.execute(
            "UPDATE memories_v8 SET importance=?, confidence=?, access_count=?, last_accessed=? WHERE id=?",
            (new_imp, int(confidence), int(acc + 1), now, mid)
        )
        con.commit()
        con.close()
        return

    cur.execute(
        """
        INSERT INTO memories_v8(scope, kind, user_id, chat_id, session_id, content, content_hash,
                               importance, confidence, access_count, created_at, last_accessed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(scope), (kind or "fact")[:24],
            user_id, chat_id, session_id,
            content, ch,
            int(max(0, min(importance, 100))),
            int(max(0, min(confidence, 100))),
            0,
            now,
            now,
        ),
    )
    con.commit()
    con.close()

def _decayed_importance(now_ts: int, created_ts: int, importance: int) -> float:
    # half-life decay: importance * 0.5^(age/half_life)
    if MEMORY_DECAY_HALFLIFE_DAYS <= 0:
        return float(importance)
    age_days = max(0.0, (now_ts - created_ts) / 86400.0)
    return float(importance) * (0.5 ** (age_days / MEMORY_DECAY_HALFLIFE_DAYS))

def _reinforcement(access_count: int) -> float:
    # diminishing returns reinforcement
    return math.log(max(1, access_count) + 1.0) * 2.5

def _confidence_term(conf: int) -> float:
    return float(max(0, min(conf, 100))) / 25.0  # 0..4

def _score_memory_row(now_ts: int, row: Dict[str, Any]) -> float:
    di = _decayed_importance(now_ts, int(row["created_at"]), int(row["importance"]))
    rf = _reinforcement(int(row["access_count"]))
    cf = _confidence_term(int(row["confidence"]))
    # prefer recently accessed slightly (keeps “working set”)
    la = int(row["last_accessed"])
    recency_access_days = max(0.0, (now_ts - la) / 86400.0)
    access_recency = 2.0 * (0.5 ** (recency_access_days / max(1.0, MEMORY_DECAY_HALFLIFE_DAYS / 2.0)))
    return di + rf + cf + access_recency

def db_fetch_candidates_v8(
    scope: int,
    user_id: Optional[str],
    chat_id: Optional[str],
    session_id: str,
    since_ts: Optional[int],
    limit: int
) -> List[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    q = "SELECT * FROM memories_v8 WHERE scope=?"
    args: List[Any] = [int(scope)]

    if since_ts is not None:
        q += " AND created_at >= ?"
        args.append(int(since_ts))

    if scope == SCOPE_PROFILE:
        if user_id:
            q += " AND user_id=?"
            args.append(user_id)
        else:
            q += " AND session_id=?"
            args.append(session_id)
    else:
        if chat_id:
            q += " AND chat_id=?"
            args.append(chat_id)
        else:
            q += " AND session_id=?"
            args.append(session_id)

    q += " ORDER BY last_accessed DESC, created_at DESC LIMIT ?"
    args.append(int(limit))

    cur.execute(q, args)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def db_reinforce_memories_v8(ids: List[int], boost: int) -> None:
    if not ids:
        return
    now = int(time.time())
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    for mid in ids:
        # Function/tool models must not serve non-tool buckets
        if ("bucket" in locals() and bucket != "tool") and FUNCTION_MODEL_PAT.search(str(mid)):
            continue
        cur.execute(
            "SELECT importance, access_count FROM memories_v8 WHERE id=?",
            (int(mid),)
        )
        row = cur.fetchone()
        if not row:
            continue
        imp = int(row[0])
        acc = int(row[1])
        new_imp = min(100, max(0, imp + int(boost)))
        cur.execute(
            "UPDATE memories_v8 SET importance=?, access_count=?, last_accessed=? WHERE id=?",
            (new_imp, acc + 1, now, int(mid))
        )
    con.commit()
    con.close()

def human_age(now_ts: int, then_ts: int) -> str:
    d = max(0, now_ts - then_ts)
    if d < 90:
        return "just now"
    if d < 3600:
        return f"{d // 60} min ago"
    if d < 86400:
        return f"{d // 3600} h ago"
    if d < 86400 * 14:
        return f"{d // 86400} days ago"
    if d < 86400 * 60:
        return f"{d // (86400 * 7)} weeks ago"
    if d < 86400 * 365:
        return f"{d // (86400 * 30)} months ago"
    return f"{d // (86400 * 365)} years ago"

def ts_to_local(ts: int) -> str:
    import datetime as _dt
    dt = _dt.datetime.fromtimestamp(ts, tz=now_dt().tzinfo)
    return dt.strftime("%Y-%m-%d %H:%M")

def build_memory_system_message_v8(
    profile_rows: List[Dict[str, Any]],
    project_rows: List[Dict[str, Any]],
    *,
    include_time: bool = False,
) -> Optional[Dict[str, str]]:
    """Build the memory system message.

    v9.1.1 policy:
    - Keep timestamps INTERNAL by default (avoid clutter + accidental date leakage into normal chat).
    - Only include timestamps when the user is explicitly asking for past recall ("when did I say...").
    """
    if not profile_rows and not project_rows:
        return None

    now_ts = int(time.time())
    parts: List[str] = [
        "Stored memory. Use it only if relevant. Do not invent or expand it.",
    ]
    if include_time:
        parts.append("Timestamps are included because the user asked about past context/time.")

    def fmt_row(r: Dict[str, Any]) -> str:
        kind = (r.get("kind") or "fact")
        imp = int(r.get("importance") or 0)
        acc = int(r.get("access_count") or 0)
        base = f"- (kind={kind}, imp={imp}, seen={acc}) {r['content']}"
        if not include_time:
            return base
        age = human_age(now_ts, int(r["created_at"]))
        loc = ts_to_local(int(r["created_at"]))
        return f"- [{age} | {loc}] (kind={kind}, imp={imp}, seen={acc}) {r['content']}"

    if profile_rows:
        parts.append("Profile memory:")
        for r in profile_rows:
            parts.append(fmt_row(r))

    if project_rows:
        parts.append("Project memory:")
        for r in project_rows:
            parts.append(fmt_row(r))

    content = "\n".join(parts).strip()
    if len(content) > MEMORY_MAX_CHARS_TOTAL:
        content = content[:MEMORY_MAX_CHARS_TOTAL] + "\n(…truncated)"
    return {"role": "system", "content": content}
# ---------------- Time-window detection ----------------
# Used for "recall from X" style requests (past lookback).
# We support both numeric and fuzzy quantities (couple/few/several) and common phrases.
TIME_PAT = re.compile(
    r"\b(\d+)\s*(min|mins|minute|minutes|h|hr|hour|hours|day|days|week|weeks|month|months|year|years)\b",
    re.I,
)
WORD_TIME_PAT = re.compile(
    r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several)\s*(?:of\s*)?"
    r"(min|mins|minute|minutes|h|hr|hour|hours|day|days|week|weeks|month|months|year|years)\b",
    re.I,
)

_WORD_NUM = {
    "a": 1, "an": 1,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2,
    "few": 3,
    "several": 6,
}

def _seconds_since_local_midnight() -> int:
    dt = now_dt()
    import datetime as _dt
    midnight = _dt.datetime(dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)
    return int((dt - midnight).total_seconds())

def parse_lookback_seconds(text: str) -> Optional[int]:
    t = (text or "").lower()

    # Common relative phrases
    if "just now" in t or "a moment ago" in t:
        return 120
    if "earlier today" in t or "today earlier" in t:
        return max(60, _seconds_since_local_midnight())
    if "this morning" in t:
        # If it's afternoon/evening, this usually means "since ~6am"
        dt = now_dt()
        import datetime as _dt
        six_am = _dt.datetime(dt.year, dt.month, dt.day, 6, 0, tzinfo=dt.tzinfo)
        return int(max(60.0, (dt - six_am).total_seconds()))
    if "last night" in t:
        # Rough: within the last 12 hours
        return 12 * 3600
    if "yesterday" in t:
        return 86400
    if "last week" in t:
        return 86400 * 7
    if "last month" in t:
        return 86400 * 30
    if "last year" in t:
        return 86400 * 365

    # Fuzzy quantities ("a few days", "couple hours", ...)
    m2 = WORD_TIME_PAT.search(t)
    if m2:
        w = m2.group(1).lower()
        unit = m2.group(2).lower()
        n = _WORD_NUM.get(w, 1)
        return _unit_to_seconds(n, unit)

    # Numeric quantities ("2 days", "90 min", ...)
    m = TIME_PAT.search(t)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    return _unit_to_seconds(n, unit)

def _unit_to_seconds(n: int, unit: str) -> int:
    if unit.startswith(("min", "minute")):
        return n * 60
    if unit.startswith(("h", "hr", "hour")):
        return n * 3600
    if unit.startswith("day"):
        return n * 86400
    if unit.startswith("week"):
        return n * 86400 * 7
    if unit.startswith("month"):
        return n * 86400 * 30
    if unit.startswith("year"):
        return n * 86400 * 365
    return 0

def user_requests_past_recall(text: str) -> bool:
    t = (text or "").lower()
    # lightweight recall detector (regex-free, low CPU)
    if any(k in t for k in [
        "remember", "remind me what", "earlier", "previous", "before",
        "last time", "we talked", "we discussed", "we said", "you said",
        "ago", "yesterday", "last week", "last month", "last year",
        "this morning", "last night",
    ]):
        return True
    return parse_lookback_seconds(t) is not None

# ---------------- Gatekeeper helpers ----------------


def gatekeeper_prompt(user_message: str) -> str:
    # v8.1.6 gatekeeper (Gemma 1B) — 4 states, JSON-only output.
    # Tightened with extra examples to reduce false positives for "Explain/Summarize/Help" requests.
    return f"""Classify the USER'S INTENT into one of 4 states.
Output ONLY valid JSON: {{"state": X}}

STATES:
0: IGNORE. General questions, greetings, or requests for explanation/help (do NOT save).
1: SAVE FACT. New permanent information about the user or project requirements.
2: DIRECTIVE. Commands on HOW the AI must behave (Always/Never/Style).
3: CORRECTION. Updating, changing, or deleting a previously stated fact.

EXAMPLES:
- "Hi" -> {{"state": 0}}
- "What is the time?" -> {{"state": 0}}
- "Explain LRU in simple terms." -> {{"state": 0}}
- "Summarize this text." -> {{"state": 0}}
- "Help me debug this error." -> {{"state": 0}}

- "The buffer size is 16MB." -> {{"state": 1}}
- "I live in Paris." -> {{"state": 1}}

- "Always use C++17." -> {{"state": 2}}
- "Never use emojis." -> {{"state": 2}}

- "Actually, use 32MB instead of 16MB." -> {{"state": 3}}
- "Forget my birthday." -> {{"state": 3}}

INPUT:
{user_message}"""



# -------- Gatekeeper prefilter (v8.1.6) --------
# Purpose: eliminate common false-positives where the gatekeeper saves "requests for explanation"
# as facts. This is intentionally conservative and aligned with state=0 definition.
# You can disable it with ROUTER_GATEKEEPER_PREFILTER=0.
ROUTER_GATEKEEPER_PREFILTER = os.getenv("ROUTER_GATEKEEPER_PREFILTER", "1").strip().lower() in ("1","true","yes","y")

_PREFILTER_IGNORE_PREFIXES = (
    "explain ",
    "describe ",
    "summarize ",
    "help me ",
    "help ",
    "debug ",
    "how do i",
    "how to",
    "what is",
    "what's",
    "why ",
    "tell me about",
)

def gatekeeper_prefilter_state(user_text: str) -> Optional[int]:
    """Return a forced gatekeeper state if obvious; else None.

    v8.1.6: Force state=0 (IGNORE) for common question/help/explanation requests to prevent
    accidental memory pollution.
    """
    if not ROUTER_GATEKEEPER_PREFILTER:
        return None
    t = (user_text or "").strip()
    if not t:
        return None
    tl = t.lower().strip()

    # question mark is a strong signal of "request"
    if tl.endswith("?"):
        return 0

    # imperatives / request prefixes
    for p in _PREFILTER_IGNORE_PREFIXES:
        if tl.startswith(p):
            return 0

    return None


# -------- Quick heuristic memory classifier (v8.1.6, optional) --------
# OFF by default for correctness. When ON, we only classify when obvious; else fall back to LLM gatekeeper.
ROUTER_QUICK_MEMORY_CLASSIFIER = os.getenv("ROUTER_QUICK_MEMORY_CLASSIFIER", "0").strip().lower() in ("1","true","yes","y")

# Lightweight signals for "fact-like" statements (state=1)
_QUICK_PROFILE_PAT = re.compile(
    r"\b(my\s+name\s+is|i\s+am\s+\d+|i\s+live\s+in|my\s+favorite|i\s+prefer|i\s+like|i\s+don't\s+like)\b",
    re.IGNORECASE,
)
_QUICK_PROJECT_PAT = re.compile(
    r"\b(buffer\s+size|page\s+size|we\s+need\s+to|requirement|constraint|must\s+implement|use\s+\d+mb|use\s+\d+kb)\b",
    re.IGNORECASE,
)

def quick_memory_state(user_text: str) -> Optional[int]:
    """Fast heuristic classifier.

    v8.1.6 states:
      0 ignore (we return None to fall back to gatekeeper)
      1 save fact
      2 directive
      3 correction
    """
    t = (user_text or "").strip()
    if not t:
        return None
    tl = t.lower()

    # Correction keywords
    if any(k in tl for k in ["actually", "change", "instead", "update", "correction", "forget", "delete", "remove"]):
        return 3

    # Directive keywords
    if any(k in tl for k in ["always", "never", "must", "do not", "don't", "style", "format", "respond in", "answer in"]):
        return 2

    if _QUICK_PROFILE_PAT.search(t) or _QUICK_PROJECT_PAT.search(t):
        return 1

    return None

def parse_gatekeeper_state(text: str, allowed: Optional[List[int]] = None) -> Optional[int]:
    """Parse gatekeeper output.

    Supports future expansion beyond 0/1/2 by passing `allowed` (from policy).
    """
    allowed = allowed or gatekeeper_allowed_states()
    allowed_set = set(allowed)

    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None

    # JSON-first (preferred)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and obj.get("state") in allowed_set:
            return int(obj["state"])
    except Exception:
        pass

    # Fallback: first integer token
    m = re.search(r"\b(-?\d+)\b", t)
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    return v if v in allowed_set else None

# Anti-thrash / governor
NODE_MODEL_STICKY_S = float(os.getenv("ROUTER_NODE_MODEL_STICKY_S", "1800"))
NODE_SWITCH_COOLDOWN_S = float(os.getenv("ROUTER_NODE_SWITCH_COOLDOWN_S", "30"))

ROUTER_MODE_NODES = [x.strip() for x in os.getenv("ROUTER_ROUTER_MODE_NODES", "B-GPU0,B-CPU0,B-PHONE0,B-PHONE2").split(",") if x.strip()]

LAT_EMA_ALPHA = float(os.getenv("ROUTER_LAT_EMA_ALPHA", "0.2"))
LAT_INIT_MS = float(os.getenv("ROUTER_LAT_INIT_MS", "1500.0"))

W_LAT = float(os.getenv("ROUTER_W_LAT", "1.0"))
W_LOAD = float(os.getenv("ROUTER_W_LOAD", "1.3"))
W_QUEUE = float(os.getenv("ROUTER_W_QUEUE", "1.6"))
W_STICKY = float(os.getenv("ROUTER_W_STICKY", "0.6"))
W_SWITCH_PENALTY = float(os.getenv("ROUTER_W_SWITCH_PENALTY", "1.0"))
W_ROUTERMODE_PENALTY = float(os.getenv("ROUTER_W_ROUTERMODE_PENALTY", "0.4"))

# ---------------- Global System Prompt ----------------
GLOBAL_SYSTEM_PROMPT = os.getenv("ROUTER_GLOBAL_SYSTEM_PROMPT", "").strip()
GLOBAL_SYSTEM_PROMPT_MODE = os.getenv("ROUTER_GLOBAL_SYSTEM_PROMPT_MODE", "prepend").strip().lower()

# ---------------- Default nodes ----------------
DEFAULT_NODES = {
    # Brain fabric naming (role/tier oriented). Hardware/topology stays internal.
    "B-PHONE0": os.getenv("BACKEND_B_PHONE0_URL", os.getenv("BACKEND_BN10_URL", "http://192.168.1.63:11436")),
    "B-PHONE1": os.getenv("BACKEND_B_PHONE1_URL", os.getenv("BACKEND_BN8_URL", "http://192.168.1.64:11437")),
    # Xperia XZ Premium (XPZ) — tool/function tier
    "B-PHONE2": os.getenv("BACKEND_B_PHONE2_URL", os.getenv("BACKEND_XPZ_URL", "http://192.168.1.65:11438")),
    "B-PHONE3": os.getenv("BACKEND_B_PHONE3_URL", os.getenv("BACKEND_BN11_URL", "http://192.168.1.66:11439")),
    "B-CPU0":   os.getenv("BACKEND_B_CPU0_URL",   os.getenv("BACKEND_PC_CPU_URL", "http://192.168.1.62:11434")),
    "B-GPU0":   os.getenv("BACKEND_B_GPU0_URL",   os.getenv("BACKEND_PC_GPU_URL", "http://192.168.1.62:11435")),


}


# ---------------- Optional extra nodes (growth-friendly) ----------------
# Add nodes without touching code:
#   ROUTER_EXTRA_NODES_JSON='{"B-PHONE3":"http://192.168.1.66:11439","B-GPU1":"http://..."}'
# or:
#   ROUTER_EXTRA_NODES_PATH='/path/to/nodes.json'
#
# Notes:
# - Values must be base URLs (scheme://host:port).
# - This merges into DEFAULT_NODES at startup.
def load_extra_nodes() -> Dict[str, str]:
    raw = (os.getenv("ROUTER_EXTRA_NODES_JSON", "") or "").strip()
    path = (os.getenv("ROUTER_EXTRA_NODES_PATH", "") or "").strip()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
        except Exception as e:
            log.warning(f"[extra_nodes] failed to read path: {e!r}")
            raw = raw
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out: Dict[str, str] = {}
            for k, v in obj.items():
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    out[k.strip()] = v.strip()
            return out
    except Exception as e:
        log.warning(f"[extra_nodes] invalid JSON; ignoring ({e!r})")
    return {}

# Merge extra nodes into defaults
try:
    _extra_nodes = load_extra_nodes()
    if _extra_nodes:
        DEFAULT_NODES.update(_extra_nodes)
except Exception:
    pass
DEFAULT_NODE_TAGS = {
    "B-PHONE1": ["phone", "mem_gate"],
    "B-PHONE0": ["phone", "chat", "tool"],
    "B-PHONE3": ["phone", "chat", "intermediate"],
    "B-GPU0":   ["pc", "gpu", "routermode"],
    "B-CPU0":   ["pc", "cpu", "routermode"],

    # Tool/function executor (XPZ)
    "B-PHONE2": ["phone", "tool"],
}

NODE_MAX_CTX_DEFAULT = {
    "B-PHONE1": 2048,
    "B-PHONE0": 2048,
    "B-PHONE2": 2048,
    "B-PHONE3": 4096,
    "B-GPU0":   8192,
    "B-CPU0":   16384,
}

def node_max_ctx(node: str) -> int:
    key = f"NODE_MAX_CTX_{node.upper().replace('-','_')}"
    v = os.getenv(key)
    if v:
        try:
            return int(v)
        except Exception:
            pass
    return NODE_MAX_CTX_DEFAULT.get(node, 8192)

DEFAULT_BUCKET_NODE_ORDER = {
    # Memory gate / classifier tier
    "mem_gate": ["B-PHONE1", "B-CPU0"],

    # Tool/function tier (XPZ first, CPU fallback)
    "tool": ["B-PHONE2", "B-CPU0"],
    "ui_helper": ["B-PHONE1", "B-CPU0"],

    "micro": ["B-PHONE0", "B-PHONE3", "B-CPU0", "B-GPU0"],
    "chat": ["B-PHONE0", "B-PHONE3", "B-CPU0", "B-GPU0"],
    "deep": ["B-GPU0", "B-CPU0", "B-PHONE3", "B-PHONE0"],
    "code": ["B-GPU0", "B-CPU0", "B-PHONE0"],
    "vision_fast": ["B-GPU0"],
    "vision_reasoning": ["B-GPU0"],
    "vision_detail_text": ["B-CPU0", "B-GPU0"],
    "health": ["B-GPU0", "B-CPU0"],
    "longctx": ["B-PHONE3", "B-CPU0", "B-GPU0", "B-PHONE0"],
}

DEFAULT_BUCKET_HINTS = {
    ("B-GPU0",   "vision_detail_text"): "Gemma-3-12B-Text",
    ("B-CPU0",   "vision_detail_text"): "Gemma-3-4B-Text",    ("B-PHONE1", "mem_gate"): "google_gemma-3-1b-it-Q8_0",
    ("B-CPU0",   "mem_gate"): "google_gemma-3-1b-it-Q8_0",

    ("B-PHONE1", "ui_helper"): "gemma3-1b-chat@2k-q8_0",
    ("B-CPU0",   "ui_helper"): "gemma3-1b-chat@2k-q8_0",

    ("B-PHONE2", "tool"): "qwen2.5-0.5b-instruct@1k-q4km",
    ("B-CPU0",   "tool"): "qwen2.5-0.5b-instruct@1k-q4km",

    ("B-PHONE0", "chat"): "gemma3-1b-chat@2k-q8_0",
    ("B-PHONE3", "chat"): "gemma3-1b-chat@4k-q8_0",
    ("B-CPU0",   "chat"): "gemma3-1b-chat@2k-q8_0",
    ("B-GPU0",   "chat"): "gemma3-1b-chat@2k-q8_0",

    ("B-GPU0",   "deep"): "Gemma-3-12B-Text",
    ("B-CPU0",   "deep"): "Gemma-3-4B-Text",

    ("B-GPU0",   "code"): "DeepSeek-R1-Distill-Llama-8B",

    ("B-GPU0",   "vision_fast"): "Gemma-3-4B-Vision",
    ("B-GPU0",   "vision_reasoning"): "Gemma-3-4B-Vision",

    ("B-GPU0",   "health"): MEDGEMMA_DEFAULT_MODEL,
    ("B-CPU0",   "health"): MEDGEMMA_DEFAULT_MODEL,
    ("B-CPU0",   "longctx"): "Qwen",

    ("B-PHONE0", "micro"): "Gemma",
    ("B-PHONE3", "micro"): "gemma3-1b-chat@4k-q8_0",
}

ROUTER_CHAT_FAILOVER_ORDER = [n.strip() for n in os.getenv("ROUTER_CHAT_FAILOVER_ORDER", "B-PHONE0,B-PHONE3,B-CPU0,B-GPU0").split(",") if n.strip()]
ROUTER_DISCOVERY_LOG_EACH_CYCLE = int(os.getenv("ROUTER_DISCOVERY_LOG_EACH_CYCLE", "1"))
ROUTER_DISCOVERY_LOG_MODELS = int(os.getenv("ROUTER_DISCOVERY_LOG_MODELS", "0"))
ROUTER_NODE_DOWN_LOG_LIMIT = int(os.getenv("ROUTER_NODE_DOWN_LOG_LIMIT", "160"))

def bucket_hint(node: str, bucket: str) -> str:
    # Vision buckets: allow dedicated preferred model IDs via env (keeps naming conventions flexible).
    if bucket == "vision_fast":
        return VISION_FAST_MODEL_ID
    if bucket == "vision_reasoning":
        return VISION_REASON_MODEL_ID
    key = f"ROUTER_HINT_{bucket.upper()}_{node.upper().replace('-','_')}"
    v = os.getenv(key, "").strip()
    if v:
        return v
    return DEFAULT_BUCKET_HINTS.get((node, bucket), "")

# ---------------- Heuristics ----------------
CODE_PAT = re.compile(r"```|traceback|Exception:|stack trace|def\s+\w+\(|class\s+\w+\(", re.IGNORECASE)
MICRO_PAT = re.compile(r"\b(rewrite|rephrase|fix|typo|correct|translate|polish|shorter)\b", re.IGNORECASE)
DEEP_PAT = re.compile(r"\b(analy[sz]e|design|architecture|plan|compare|derive|prove|reason)\b", re.IGNORECASE)
VISION_REASON_PAT = re.compile(r"\b(why|analy[sz]e|explain|compare|diagnos|interpret|root cause|meaning)\b", re.IGNORECASE)

# Tool-selection (OpenWebUI / tool-router meta prompts)
FUNCTION_MODEL_PAT = re.compile(r"\bfunctiongemma\b|\btool@\b|\b-function\b", re.IGNORECASE)
# v33.2 (name-logic): model identity resolution ladder
# Prefer cluster aliases (stable names). Use vendor names only as compatibility fallback.
MODEL_VENDOR_FALLBACK = {
    # alias -> vendor
    "gemma3-1b-chat@2k-q8_0": "google_gemma-3-1b-it-Q8_0",
    "gemma3-1b-it@2k-q8_0": "google_gemma-3-1b-it-Q8_0",
    "gemma3-1b-it@1k-q8_0": "google_gemma-3-1b-it-Q8_0",
}
MODEL_ALIAS_CANONICAL = {
    # vendor -> alias
    "google_gemma-3-1b-it-Q8_0": "gemma3-1b-chat@2k-q8_0",
}

def resolve_model_name_for_node(req_id: str, node, requested: str, bucket_default: str):
    """Resolve model name for a specific node.

    Ladder:
      1) Use requested name if node advertises it
      2) If requested is vendor name, map to alias (vendor->alias) and try
      3) If requested is alias, map to vendor (alias->vendor) and try
      4) Fallback to bucket_default if node advertises it
      5) Else return requested unchanged (caller can handle failure)
    """
    try:
        advertised = set(getattr(node, "models", {}) or {})
    except Exception:
        advertised = set()
    if not advertised:
        return requested

    def _has(mid: str) -> bool:
        return mid in advertised

    if requested and _has(requested):
        return requested

    cand = MODEL_ALIAS_CANONICAL.get(str(requested))
    if cand and _has(cand):
        try:
            log.info(f"[{req_id}] model_resolve vendor->alias {requested} -> {cand} node={getattr(node,'name',None)}")
        except Exception:
            pass
        return cand

    cand = MODEL_VENDOR_FALLBACK.get(str(requested))
    if cand and _has(cand):
        try:
            log.info(f"[{req_id}] model_resolve alias->vendor {requested} -> {cand} node={getattr(node,'name',None)}")
        except Exception:
            pass
        return cand

    if bucket_default and _has(bucket_default):
        try:
            log.info(f"[{req_id}] model_resolve fallback_default {requested} -> {bucket_default} node={getattr(node,'name',None)}")
        except Exception:
            pass
        return bucket_default

    return requested

TOOL_SELECT_PAT = re.compile(
    r"\bavailable\s+tools\b|\btool_calls\b|\byour\s+task\s+is\s+to\s+choose\s+and\s+return\s+the\s+correct\s+tool",
    re.IGNORECASE,
)

OPS_HEALTH_PAT = re.compile(
    r"\b(health\s*check|healthcheck|/health|liveness|readiness|probe|monitor(ing)?|"
    r"status\s*endpoint|service\s*status|uptime|latency|error\s*rate|metrics|"
    r"logs?|traces?|router|load\s*balanc|uvicorn|fastapi|http\s*1\.1|status\s*code|"
    r"system\s*health|service\s*health|router\s*health|node\s*health|cluster\s*health|"
    r"database\s*health|db\s*health|api\s*health|script\s*health|python\s*script|server\s*health|"
    r"cpu|gpu|ram|memory|disk|filesystem|network|postgres|mysql|sqlite|redis|mongodb|llama\.cpp|docker|container)\b",
    re.IGNORECASE,
)
HEALTH_PAT = re.compile(
    r"\b(medical|medicine|clinical|patient|doctor|nurse|hospital|clinic|radiology|radiologist|"
    r"diagnos(?:is|e|tic)?|symptom|symptoms|treatment|medication|prescription|disease|"
    r"tumou?r|lesion|fracture|infection|pneumonia|cancer|stroke|hemorrhage|haemorrhage|"
    r"x[- ]?ray|radiograph|ct|ct\s*scan|scanner|mri|irm|pet\s*scan|ultrasound|ecg|eeg|biopsy)\b",
    re.IGNORECASE,
)

SCCI_MEDICAL_DEMO_DISCLAIMER_IN = (
    "Medical Demo Notice: This system is a research and demonstration prototype. "
    "Outputs are for educational image description and technical evaluation only and are NOT medical diagnosis. "
    "Clinical interpretation and decision-making must be performed by qualified healthcare professionals."
)

SCCI_MEDICAL_DEMO_DISCLAIMER_OUT = (
    "⚠️ Medical Demo Disclaimer\n"
    "This output is for research, education, and technical demonstration only. "
    "It is NOT a medical diagnosis, treatment recommendation, or clinical decision support. "
    "Any interpretation must be reviewed by a qualified healthcare professional.\n\n"
)

def _is_medical_intent_text(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if OPS_HEALTH_PAT.search(t):
        return False
    return bool(HEALTH_PAT.search(t))


# ---------------- SCCI Priority Policy Engine ----------------
SCCI_PRIORITY_POLICY_ENABLE = int(os.getenv("SCCI_PRIORITY_POLICY_ENABLE", "1"))

def _scci_priority_policies() -> list[dict]:
    """Ordered first-match-wins routing policies."""
    return [
        {
            "name": "medical_vision",
            "priority": 100,
            "match": lambda ctx: bool(ctx.get("has_image")) and bool(ctx.get("medical_intent")),
            "action": {
                "bucket": "health",
                "node": MEDGEMMA_DEFAULT_NODE,
                "model": MEDGEMMA_DEFAULT_MODEL,
            },
        },
        {
            "name": "generic_vision",
            "priority": 80,
            "match": lambda ctx: bool(ctx.get("has_image")) and str(ctx.get("bucket") or "").startswith("vision"),
            "action": {},
        },
        {
            "name": "default_chat",
            "priority": 10,
            "match": lambda ctx: True,
            "action": {},
        },
    ]

def _evaluate_scci_priority_policy(ctx: dict) -> tuple[str | None, int | None, dict]:
    if not SCCI_PRIORITY_POLICY_ENABLE:
        return (None, None, {})
    for pol in sorted(_scci_priority_policies(), key=lambda p: int(p.get("priority", 0)), reverse=True):
        try:
            if bool(pol.get("match")(ctx)):
                return (str(pol.get("name") or ""), int(pol.get("priority", 0)), dict(pol.get("action") or {}))
        except Exception:
            continue
    return (None, None, {})

def _apply_medical_output_disclaimer_text(text: str) -> str:
    s = str(text or "")
    if not s:
        return SCCI_MEDICAL_DEMO_DISCLAIMER_OUT.rstrip()
    if s.startswith("⚠️ Medical Demo Disclaimer"):
        return s
    return SCCI_MEDICAL_DEMO_DISCLAIMER_OUT + s

def _apply_medical_output_disclaimer_json_bytes(raw: bytes) -> bytes:
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
        choices = data.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    msg["content"] = _apply_medical_output_disclaimer_text(msg.get("content") or "")
                    break
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    except Exception:
        return raw

def _medical_disclaimer_sse_chunk(req_id: str, model_id: str) -> bytes:
    created = int(time.time())
    chunk = {
        "id": f"chatcmpl-medical-demo-{req_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": str(model_id or "medical-demo"),
        "choices": [{"index": 0, "delta": {"content": SCCI_MEDICAL_DEMO_DISCLAIMER_OUT}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")

# ---------------- Helpers ----------------
def make_req_id() -> str:
    return hashlib.sha1(f"{time.time_ns()}:{os.getpid()}".encode("utf-8")).hexdigest()[:10]

def _hmac32(text: str) -> str:
    dig = hmac.new(SESSION_HMAC_KEY.encode("utf-8"), text.encode("utf-8"), hashlib.sha256).hexdigest()
    return dig[:32]

def approx_text_tokens(messages: List[Dict[str, Any]]) -> int:
    total_chars = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and (part.get("type") or "").lower() in ("text", "input_text"):
                    total_chars += len(part.get("text", ""))
        else:
            total_chars += len(str(c))
    return max(1, int(total_chars / CHARS_PER_TOKEN))

# --- Token estimation + long-paste utilities ---
def _last_user_text(messages: list) -> str:
    for m in reversed(messages or []):
        if (m or {}).get("role") == "user":
            c = (m or {}).get("content", "")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text",""))
                return "\n".join([x for x in parts if x])
            if isinstance(c, str):
                return c
    return ""

def _looks_like_summarize_request(messages: list) -> bool:
    txt = _last_user_text(messages).lower()
    return any(k in txt for k in ("summarize", "summary", "résume", "resume", "tl;dr", "tldr", "too long", "too-long", "summarise", "résumer", "resumer"))

def _estimate_tokens_from_messages(messages: list) -> int:
    total_chars = 0
    for m in (messages or []):
        c = (m or {}).get("content", "")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    total_chars += len(p.get("text","") or "")
                elif isinstance(p, dict) and p.get("type") == "image_url":
                    total_chars += 1200
    return int(total_chars / max(1.0, ROUTER_TOKEN_EST_CHARS_PER_TOKEN))



def _select_longtext_ladder_from_final_body(
    req_id: str,
    body: dict,
    fallback_msgs: list,
    fallback_tokens: int,
):
    """
    Restore the missing longtext selector used by the v11.x long-paste path.
    Decide the ladder from the exact outbound payload after rebuild/rewrite.
    """
    try:
        final_msgs = body.get("messages") or fallback_msgs or []
        final_tokens = _estimate_tokens_from_messages(final_msgs) + int(ROUTER_CTX_COMPLETION_RESERVE)
        chosen_tokens = max(int(fallback_tokens or 0), int(final_tokens or 0))
        chat_target_model = _pick_chat_ladder_model(int(chosen_tokens or 0))
        chat_target_node = _pick_chat_ladder_node(chat_target_model)
        chat_target_ctx = int(_parse_ctx_from_model_id(chat_target_model) or 4096)
        chat_ratio = float(int(chosen_tokens or 0)) / float(max(1, chat_target_ctx))

        if int(chosen_tokens or 0) >= int(LONGTEXT_EMERGENCY_TOKENS or 12000):
            valid = [m for m in CHAT_LADDER_MODELS if _parse_ctx_from_model_id(m) > 0]
            if valid:
                largest_model = max(valid, key=_parse_ctx_from_model_id)
                largest_node = _pick_chat_ladder_node(largest_model)
                largest_ctx = int(_parse_ctx_from_model_id(largest_model) or chat_target_ctx or 4096)
                largest_ratio = float(int(chosen_tokens or 0)) / float(max(1, largest_ctx))
                log.warning(
                    f"[{req_id}] → SCCI LONGTEXT_EMERGENCY_ESCALATION sid={req_id} "
                    f"chosen_tokens~{int(chosen_tokens or 0)} threshold={int(LONGTEXT_EMERGENCY_TOKENS or 12000)} "
                    f"target={largest_node} model={largest_model}"
                )
                chat_target_model = largest_model
                chat_target_node = largest_node
                chat_target_ctx = largest_ctx
                chat_ratio = largest_ratio

        try:
            body["_router_force_model"] = chat_target_model
            body["_router_force_node"] = chat_target_node
            body["_router_force_reason"] = "longtext_final_rebuild"
        except Exception:
            pass

        log.info(
            f"[{req_id}] → SCCI FINAL_REBUILD_CTX sid={req_id} "
            f"buffer_ratio={float(LONGTEXT_ESCALATION_BUFFER_RATIO or 1.25):.2f} "
            f"fallback_tokens~{int(fallback_tokens or 0)} final_tokens~{int(final_tokens or 0)} "
            f"chosen_tokens~{int(chosen_tokens or 0)} target={chat_target_node} model={chat_target_model}"
        )
        return chosen_tokens, chat_target_model, chat_target_node, chat_target_ctx, chat_ratio
    except Exception as long_ctx_fix_err:
        log.warning(f"[{req_id}] SCCI final_rebuild_ctx_err={type(long_ctx_fix_err).__name__}:{long_ctx_fix_err}")
        chat_target_model = _pick_chat_ladder_model(int(fallback_tokens or 0))
        chat_target_node = _pick_chat_ladder_node(chat_target_model)
        chat_target_ctx = int(_parse_ctx_from_model_id(chat_target_model) or 4096)
        chat_ratio = float(int(fallback_tokens or 0)) / float(max(1, chat_target_ctx))
        return int(fallback_tokens or 0), chat_target_model, chat_target_node, chat_target_ctx, chat_ratio

_CTX_TAG_RE = re.compile(r"@(\d+)\s*k", re.IGNORECASE)

def _ctx_from_model_id(model_id: str, default_ctx: int = 4096) -> int:
    if not model_id:
        return default_ctx
    m = _CTX_TAG_RE.search(str(model_id))
    if m:
        try:
            return int(m.group(1)) * 1024
        except Exception:
            return default_ctx
    return default_ctx

def _chunk_text(s: str, chunk_chars: int, max_chunks: int) -> list[str]:
    s = s or ""
    if len(s) <= chunk_chars:
        return [s]
    chunks, i = [], 0
    while i < len(s) and len(chunks) < max_chunks:
        chunks.append(s[i:i+chunk_chars])
        i += chunk_chars
    return chunks

async def _call_nonstream_chat(client, base_url: str, body: dict, timeout_s: float = 180.0):
    b = dict(body)
    b["stream"] = False
    return await client.post(f"{base_url}/v1/chat/completions", json=b, timeout=timeout_s)

async def _summarize_longpaste_via_chunks(client, base_url: str, model_id: str, messages: list, sys_prefix: str = "") -> str:
    user_txt = _last_user_text(messages)
    chunks = _chunk_text(user_txt, ROUTER_LONGPASTE_CHUNK_CHARS, ROUTER_LONGPASTE_MAX_CHUNKS)
    summaries = []
    for idx, ch in enumerate(chunks, start=1):
        sys = sys_prefix or "You are a helpful assistant that summarizes text."
        prompt = f"Chunk {idx}/{len(chunks)}:\n{ch}\n\nWrite a concise summary of this chunk (3-6 bullet points)."
        body = {"model": model_id, "messages": [{"role":"system","content":sys},{"role":"user","content":prompt}], "stream": False}
        r = await _call_nonstream_chat(client, base_url, body)
        if r.status_code >= 400:
            # v30: ctx overflow mitigation for huge summarize requests (chunk-map-reduce on CPU chat model)
            if ROUTER_LONGPASTE_ENABLE and bucket in ("chat","deep"):
                try:
                    body_text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    if "exceeds the available context size" in body_text.lower() and _looks_like_summarize_request(body.get("messages", [])):
                        user_txt = _last_user_text(body.get("messages", []))
                        if len(user_txt) >= ROUTER_LONGPASTE_CHAR_THRESHOLD:
                            base2 = (NODES.get("B-CPU0") or NODES.get(node_name) or {}).get("url")
                            if base2:
                                cpu_models = NODE_MODELS_CACHE.get("B-CPU0") or []
                                cand = []
                                for mid in cpu_models:
                                    if FUNCTION_MODEL_PAT.search(str(mid)):
                                        continue
                                    if VISION_MODEL_PAT.search(str(mid)):
                                        continue
                                    cand.append(str(mid))
                                model2 = model_id
                                if cand:
                                    cand.sort(key=lambda x: _ctx_from_model_id(x, 4096), reverse=True)
                                    model2 = cand[0]
                                log.warning(f"[{req_id}] ctx_overflow_detected: chunk summarization on CPU0 model={model2}")
                                summary = await _summarize_longpaste_via_chunks(client, base2, model2, body.get("messages", []))
                                out = {
                                    "id": f"chatcmpl-{req_id}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": model2,
                                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": summary}}],
                                }
                                raw_out = json.dumps(out).encode("utf-8")
                                if ROUTER_DEBUG_HEADERS:
                                    try:
                                        req.state.router_bucket = "chat"
                                        req.state.router_node = "B-CPU0"
                                        req.state.router_model = model2
                                        req.state.router_reason = "ctx_overflow_chunk_summarize"
                                        req.state.router_capability_level = "3"
                                    except Exception:
                                        pass
                                return Response(content=raw_out, status_code=200, media_type="application/json")
                except Exception as e:
                    log.error(f"[{req_id}] ctx_overflow_chunk_summarize_failed: {e}")
            raise RuntimeError(f"chunk_summarize_failed status={r.status_code} body={r.text[:500]}")
        data = r.json()
        summaries.append(((data.get("choices") or [{}])[0].get("message") or {}).get("content","").strip())
    merged = "\n\n".join([s for s in summaries if s])
    sys2 = sys_prefix or "You are a helpful assistant that summarizes text."
    body2 = {"model": model_id, "messages": [{"role":"system","content":sys2},{"role":"user","content":"Combine and summarize these chunk summaries into a coherent overall summary with:\n- 6-10 bullet points\n- 1 short paragraph conclusion\n\nChunk summaries:\n"+merged}], "stream": False}
    r2 = await _call_nonstream_chat(client, base_url, body2)
    if r2.status_code >= 400:
        raise RuntimeError(f"reduce_summarize_failed status={r2.status_code} body={r2.text[:500]}")
    data2 = r2.json()
    return (((data2.get("choices") or [{}])[0].get("message") or {}).get("content","").strip())



def count_images(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                continue
            t = (part.get("type") or "").lower()
            if t in ("image_url", "input_image"):
                iu = part.get("image_url")
                if isinstance(iu, dict) and isinstance(iu.get("url"), str) and iu.get("url").strip():
                    n += 1
                elif isinstance(iu, str) and iu.strip():
                    n += 1
                elif isinstance(part.get("url"), str) and part.get("url").strip():
                    n += 1
    return n


def count_images_in_last_user_message(messages: List[Dict[str, Any]]) -> int:
    # Only count image items in the *last* user message to avoid OpenWebUI history carrying images forward.
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            n = 0
            for it in c:
                if isinstance(it, dict) and it.get("type") in ("image_url", "input_image"):
                    n += 1
            return n
        return 0
    return 0


def extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            chunks = []
            for part in c:
                if isinstance(part, dict) and (part.get("type") or "").lower() in ("text", "input_text"):
                    chunks.append(part.get("text", ""))
            return "\n".join([x for x in chunks if x])
    return ""

def get_max_tokens(body: Dict[str, Any]) -> int:
    mt = body.get("max_tokens")
    if mt is None:
        mt = body.get("max_completion_tokens")
    try:
        mt = int(mt) if mt is not None else DEFAULT_MAX_TOKENS
    except Exception:
        mt = DEFAULT_MAX_TOKENS
    return max(1, mt)

def apply_global_system_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not GLOBAL_SYSTEM_PROMPT:
        return messages
    for m in messages:
        if m.get("role") == "system" and str(m.get("content", "")).strip() == GLOBAL_SYSTEM_PROMPT:
            return messages
    sys_msg = {"role": "system", "content": GLOBAL_SYSTEM_PROMPT}
    mode = (GLOBAL_SYSTEM_PROMPT_MODE or "prepend").lower()
    if mode == "only_if_missing":
        has_system = any(m.get("role") == "system" for m in messages)
        return messages if has_system else [sys_msg] + messages
    if mode == "append":
        i = 0
        while i < len(messages) and messages[i].get("role") == "system":
            i += 1
        return messages[:i] + [sys_msg] + messages[i:]
    return [sys_msg] + messages

# ---------------- Model tagging ----------------
def is_vision_model(model_id: str) -> bool:
    s = model_id.lower()
    return ("vision" in s) or ("mmproj" in s)

def score_bucket(model_id: str) -> Dict[str, int]:
    s = model_id.lower()
    scores = {b: 0 for b in DEFAULT_BUCKET_NODE_ORDER.keys()}
    if "med" in s or "health" in s:
        scores["health"] += 50
    if is_vision_model(model_id):
        scores["vision_fast"] += 40
        scores["vision_reasoning"] += 35
        if "med" in s:
            scores["health"] += 20
    if "deepseek" in s or "r1" in s or "reason" in s or "think" in s:
        scores["deep"] += 45
        scores["code"] += 25
    if "code" in s or "coder" in s:
        scores["code"] += 45
    # Tool/function executor models
    if any(x in s for x in ("function", "tool", "tools")):
        scores["tool"] += 60
    if "32k" in s or "64k" in s or "long" in s:
        scores["longctx"] += 35
    if "instruct" in s or "chat" in s or "text" in s or "gemma" in s or "qwen" in s:
        scores["chat"] += 25
    if any(x in s for x in ("270m", "500m", "1.5b", "1.7b", "2b", "tiny", "mini")):
        scores["micro"] += 20
        scores["chat"] += 10
    return scores

# ---------------- Data structures ----------------
@dataclass
class ModelInfo:
    id: str
    vision: bool
    bucket_scores: Dict[str, int] = field(default_factory=dict)

@dataclass
class NodeMetrics:
    active_requests: int = 0
    queue_depth: int = 0
    loading: bool = False
    tokens_per_s: float = 0.0
    last_updated: float = 0.0

@dataclass
class Node:
    name: str
    base_url: str
    tags: List[str] = field(default_factory=list)

    last_ok: float = 0.0
    last_fail: float = 0.0
    last_checked: float = 0.0
    status: str = "unknown"
    fail_streak: int = 0

    ready: bool = True
    last_ready_checked: float = 0.0
    ready_status: str = "unknown"

    ready_supported: Optional[bool] = None
    metrics_supported: Optional[bool] = None

    models: Dict[str, ModelInfo] = field(default_factory=dict)

    last_model_by_bucket: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    last_bucket: Tuple[str, float] = ("", 0.0)

    metrics: NodeMetrics = field(default_factory=NodeMetrics)
    lat_ema_ms: float = float(os.getenv("ROUTER_LAT_INIT_MS", "1500.0"))
    lat_ema_by_bucket: Dict[str, float] = field(default_factory=dict)

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def in_cooldown(self) -> bool:
        return (time.time() - self.last_fail) < FAIL_COOLDOWN_S

    def healthy_cached(self) -> bool:
        return (time.time() - self.last_ok) <= HEALTH_TTL_S and not self.in_cooldown()

    def metrics_fresh(self) -> bool:
        return (time.time() - self.metrics.last_updated) <= METRICS_TTL_S

# ---------------- Globals ----------------
_nodes: Dict[str, Node] = {}
_session_map: Dict[str, Dict[str, Any]] = {}
_session_lock = asyncio.Lock()
ROUTER_STARTED_AT = time.time()
_NODE_FIRST_UP_TS: Dict[str, float] = {}

# ---------------- HTTP clients ----------------
client = httpx.AsyncClient(
    timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
)
stream_client = httpx.AsyncClient(
    timeout=httpx.Timeout(STREAM_READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
)

async def send_upstream_stream(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> httpx.Response:
    req = stream_client.build_request("POST", url, headers=headers, json=payload)
    return await stream_client.send(req, stream=True)

def passthrough_headers(req: Request) -> Dict[str, str]:
    allow = {
        "content-type", "accept", "user-agent",
        "x-request-id", "x-session-id", "x-role", "x-bucket",
        "x-force-node", "x-force-model",
        "x-user-id", "x-chat-id", "x-openwebui-user-id", "x-openwebui-chat-id", "x-conversation-id",
    }
    return {k: v for k, v in req.headers.items() if k.lower() in allow}

# ---------------- Health + readiness ----------------
async def check_node_health(node: Node) -> bool:
    now = time.time()
    if (now - node.last_checked) < HEALTH_MIN_INTERVAL_S and node.status != "unknown":
        return node.healthy_cached()

    node.last_checked = now
    if node.in_cooldown():
        node.status = "cooldown"
        return False

    last_err = None
    for path in HEALTH_PATHS:
        try:
            r = await client.get(node.url(path))
            if r.status_code < 400:
                node.last_ok = time.time()
                node.status = "ok"
                node.fail_streak = 0
                return True
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = repr(e)

    node.last_fail = time.time()
    node.status = f"fail({last_err})"
    node.fail_streak += 1
    return False

async def check_node_ready(node: Node) -> bool:
    # Optional: skip /ready probing entirely (cleaner logs for backends that don't support it).
    # Default is disabled (ROUTER_USE_READY=0).
    if not ROUTER_USE_READY:
        ok = node.healthy_cached() or await check_node_health(node)
        node.ready = ok
        node.ready_status = "disabled(use_health)" if ok else "disabled(use_health)_down"
        return ok

    if node.ready_supported is False:
        ok = node.healthy_cached() or await check_node_health(node)
        node.ready = ok
        node.ready_status = "fallback_health(no_ready_endpoint)"
        return ok

    now = time.time()
    if (now - node.last_ready_checked) < HEALTH_MIN_INTERVAL_S and node.ready_status != "unknown":
        return node.ready

    node.last_ready_checked = now

    if node.in_cooldown():
        node.ready = False
        node.ready_status = "cooldown"
        return False

    try:
        r = await client.get(node.url(READY_PATH))
        if r.status_code < 400:
            node.ready_supported = True
            try:
                data = r.json()
                ready = bool(data.get("ready", data.get("ok", True)))
                node.ready = ready
                node.ready_status = "ready" if ready else "not_ready"
                return ready
            except Exception:
                node.ready = True
                node.ready_status = "ready(200)"
                return True

        if r.status_code == 404:
            node.ready_supported = False
            ok = node.healthy_cached() or await check_node_health(node)
            node.ready = ok
            node.ready_status = "fallback_health(no_ready_endpoint)"
            return ok

        node.ready_supported = True
        node.ready = False
        node.ready_status = f"ready_http_{r.status_code}"
        return False

    except Exception:
        ok = node.healthy_cached() or await check_node_health(node)
        node.ready = ok
        node.ready_status = "fallback_health_exc"
        return ok

# ---------------- Metrics polling ----------------
async def update_node_metrics(node: Node) -> None:
    if node.metrics_supported is False:
        return

    now = time.time()
    if (now - node.metrics.last_updated) < METRICS_MIN_INTERVAL_S and node.metrics.last_updated > 0:
        return

    try:
        r = await client.get(node.url(METRICS_PATH))
        if r.status_code in (400, 404):
            node.metrics_supported = False
            node.metrics.last_updated = now
            return

        if r.status_code >= 400:
            node.metrics_supported = True
            node.metrics.last_updated = now
            return

        node.metrics_supported = True
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        node.metrics.active_requests = int(data.get("active_requests", node.metrics.active_requests or 0))
        node.metrics.queue_depth = int(data.get("queue_depth", data.get("queue", node.metrics.queue_depth or 0)))
        node.metrics.loading = bool(data.get("loading", False))
        node.metrics.tokens_per_s = float(data.get("tokens_per_s", data.get("tps", node.metrics.tokens_per_s or 0.0)))
        node.metrics.last_updated = now

    except Exception:
        node.metrics.last_updated = now

# ---------------- Discovery ----------------
async def discover_models_for_node(node: Node) -> None:
    try:
        r = await client.get(node.url(OPENAI_MODELS_PATH))
        if r.status_code >= 400:
            return
        data = r.json()
        items = data.get("data") or []
        new_models: Dict[str, ModelInfo] = {}
        for it in items:
            mid = str(it.get("id", "")).strip()
            # v32-nodefault: ignore llama.cpp placeholder alias
            if not mid or str(mid).strip() == "default":
                continue
            if not mid:
                continue
            new_models[mid] = ModelInfo(id=mid, vision=is_vision_model(mid), bucket_scores=score_bucket(mid))
        node.models = new_models
    except Exception:
        return

async def discovery_loop() -> None:
    cycle = 0

    async def _run_cycle() -> None:
        nonlocal cycle
        cycle += 1
        snapshot = []
        for node in _nodes.values():
            ok = node.healthy_cached() or await check_node_health(node)
            if ok:
                if node.name not in _NODE_FIRST_UP_TS:
                    _NODE_FIRST_UP_TS[node.name] = time.time()
                await discover_models_for_node(node)
            await update_node_metrics(node)
            ready = await check_node_ready(node)
            model_count = len(getattr(node, "models", {}) or {})
            snapshot.append(
                f"{node.name}:up={1 if ok else 0},ready={1 if ready else 0},status={node.status},ready_status={node.ready_status},models={model_count},q={int(getattr(node.metrics, 'queue_depth', 0) or 0)},act={int(getattr(node.metrics, 'active_requests', 0) or 0)}"
            )
        if ROUTER_DISCOVERY_LOG_EACH_CYCLE:
            msg = " | ".join(snapshot)
            _discovery_log("INFO", f"[DISCOVERY] cycle={cycle} nodes={len(snapshot)} {msg}")
            if ROUTER_DISCOVERY_LOG_MODELS:
                for node in _nodes.values():
                    mids = sorted(list((getattr(node, 'models', {}) or {}).keys()))
                    log.info(f"[DISCOVERY_MODELS] node={node.name} models={mids}")

    await _run_cycle()
    while True:
        await asyncio.sleep(DISCOVERY_INTERVAL_S)
        await _run_cycle()

# ---------------- Session pinning ----------------
def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items or []:
        if not it or it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out

def _fallback_order_for_bucket(bucket: str, primary_node_name: str) -> List[str]:
    if bucket in ("chat", "micro", "longctx"):
        base = list(ROUTER_CHAT_FAILOVER_ORDER)
    elif bucket == "tool":
        base = [primary_node_name, "B-CPU0"]
    elif bucket in ("vision_fast", "vision_reasoning", "vision_detail_text", "health"):
        base = [primary_node_name, "B-GPU0", "B-CPU0"]
    elif bucket in ("code", "deep"):
        base = [primary_node_name, "B-CPU0", "B-GPU0", "B-PHONE3"]
    else:
        base = [primary_node_name, "B-PHONE3", "B-CPU0", "B-GPU0"]
    return _dedupe_keep_order([primary_node_name] + base + list(bucket_node_order(bucket)))

def _mark_node_transport_down(node: Node, req_id: str, exc: Exception) -> None:
    node.last_fail = time.time()
    node.fail_streak = int(getattr(node, 'fail_streak', 0) or 0) + 1
    node.ready = False
    node.ready_status = f"transport_fail:{type(exc).__name__}"
    err = str(exc or '')
    if len(err) > ROUTER_NODE_DOWN_LOG_LIMIT:
        err = err[:ROUTER_NODE_DOWN_LOG_LIMIT] + '…'
    node.status = f"transport_fail({type(exc).__name__}:{err})"
    log.warning(f"[{req_id}] NODE_DOWN node={node.name} status={node.status} fail_streak={node.fail_streak}")

def _resolve_fallback_model(req_id: str, node: Node, bucket: str, requested_model: str) -> Optional[str]:
    try:
        if getattr(node, 'models', None) is None:
            return None
        if requested_model in getattr(node, 'models', {}):
            return requested_model
        resolved = resolve_model_name_for_node(req_id, node, requested_model, requested_model)
        if resolved in getattr(node, 'models', {}):
            return resolved
        hint = bucket_hint(node.name, bucket)
        if hint:
            hinted = find_model_by_hint(node, hint, bucket)
            if hinted:
                return hinted
        need_vis = bucket in ("vision_fast", "vision_reasoning", "vision_detail_text", "health")
        return pick_best_for_bucket(node, bucket, need_vision=need_vis)
    except Exception:
        return None

async def _pick_transport_fallback(
    req_id: str,
    bucket: str,
    failed_node_name: str,
    requested_model: str,
    tried: List[str],
    prompt_tok_est: int = 0,
    max_tokens: int = 0,
) -> Tuple[Optional[str], Optional[Node], Optional[str]]:
    need_ctx_guard = int(prompt_tok_est or 0) > 0 and int(max_tokens or 0) >= 0
    for cand_name in _fallback_order_for_bucket(bucket, failed_node_name):
        if cand_name == failed_node_name:
            continue
        cand = _nodes.get(cand_name)
        if not cand:
            tried.append(f"{cand_name}(missing)")
            continue
        if need_ctx_guard:
            try:
                if not ctx_fits(cand_name, int(prompt_tok_est or 0), int(max_tokens or 0)):
                    tried.append(
                        f"{cand_name}(ctx_insufficient:{int(prompt_tok_est or 0)}+{int(max_tokens or 0)}>{int(node_max_ctx(cand_name) * CTX_HEADROOM_RATIO)})"
                    )
                    continue
            except Exception:
                pass
        try:
            ok = await check_node_health(cand)
        except Exception as e:
            tried.append(f"{cand_name}(health_exc:{type(e).__name__})")
            continue
        if not ok:
            tried.append(f"{cand_name}(unhealthy:{cand.status})")
            continue
        try:
            ready = await check_node_ready(cand)
        except Exception as e:
            tried.append(f"{cand_name}(ready_exc:{type(e).__name__})")
            continue
        if not ready:
            tried.append(f"{cand_name}(not_ready:{cand.ready_status})")
            continue
        try:
            await discover_models_for_node(cand)
        except Exception:
            pass
        cand_model = _resolve_fallback_model(req_id, cand, bucket, requested_model)
        if not cand_model:
            tried.append(f"{cand_name}(no_model_for_{bucket})")
            continue
        if need_ctx_guard:
            try:
                model_ctx = int(_parse_ctx_from_model_id(cand_model) or 0)
                if model_ctx > 0 and (int(prompt_tok_est or 0) + int(max_tokens or 0)) > int(model_ctx * CTX_HEADROOM_RATIO):
                    tried.append(
                        f"{cand_name}(model_ctx_insufficient:{cand_model}:{int(prompt_tok_est or 0)}+{int(max_tokens or 0)}>{int(model_ctx * CTX_HEADROOM_RATIO)})"
                    )
                    continue
            except Exception:
                pass
        return cand_name, cand, cand_model
    return None, None, None

def extract_session_id(req: Request, body: Dict[str, Any]) -> str:
    """Derive a stable session id for pinning/stickiness.

    Clean gateway rule:
      1) session_id = chat_id (conversation id)
      2) else session_id = user_id
      3) else session_id = fallback HMAC
    """
    # Explicit override (lets you force a stable session from a client/proxy)
    sid = (req.headers.get("x-session-id") or "").strip()
    if sid:
        return sid

    # 1) Prefer chat/conversation id (OpenWebUI usually has this when enabled)
    chat_id = extract_chat_id(req, body)
    if chat_id:
        return str(chat_id)

    # 2) Otherwise fall back to user id
    user_id = extract_user_id(req, body)
    if user_id:
        return str(user_id)

    # 3) Last resort: deterministic HMAC over first user message (or body)
    msgs = body.get("messages") or []
    msgs = apply_global_system_prompt(msgs)
    body["messages"] = msgs

    first_user = ""
    for m in msgs:
        if m.get("role") == "user":
            first_user = str(m.get("content", ""))[:512]
            break
    if not first_user:
        first_user = json.dumps(body, sort_keys=True)[:512]
    return _hmac32(first_user)

async def get_session_state(session_id: str) -> Dict[str, Any]:
    """Return a mutable session state dict (created on first use).

    v9 change:
    - We keep *capability memory* (level 1/2/3) per session.
    - We keep a pinned (node, model) but we only reuse it if it still fits health+ctx.
    - We do NOT store to disk (KISS). TTL cleans up old sessions.
    """
    now = time.time()
    async with _session_lock:
        s = _session_map.get(session_id)
        if not s:
            s = {
                "capability_level": max(1, min(3, int(CAP_LEVEL_DEFAULT))),
                "pinned_node": None,
                "pinned_model": None,
                "pinned_capability": None,
                "pinned_at": 0.0,
                "simple_streak": 0,
                "last_tech_ts": 0.0,
                "updated_at": now,
                "session_id": session_id,
                "lane": "chat",
                "bearer_state": "IDLE",
                "current_node": None,
                "current_model": None,
                "current_action": "none",
                "current_cause": "",
                "prewarmed_target": None,
                "prewarmed_model": None,
                "handover_pending": 0,
                "bearer_locked": 0,
                "last_ctx_needed": 0,
                "last_ctx_limit": 0,
                "last_ctx_ratio": 0.0,
            }
            _session_map[session_id] = s
            return s

        # TTL cleanup: if session is too old, reset it (same as v8 behavior, but for full state).
        if (now - float(s.get("updated_at", 0.0))) > float(SESSION_TTL_S):
            _session_map.pop(session_id, None)
            s = {
                "capability_level": max(1, min(3, int(CAP_LEVEL_DEFAULT))),
                "pinned_node": None,
                "pinned_model": None,
                "pinned_capability": None,
                "pinned_at": 0.0,
                "simple_streak": 0,
                "last_tech_ts": 0.0,
                "updated_at": now,
                "session_id": session_id,
                "lane": "chat",
                "bearer_state": "IDLE",
                "current_node": None,
                "current_model": None,
                "current_action": "none",
                "current_cause": "",
                "prewarmed_target": None,
                "prewarmed_model": None,
                "handover_pending": 0,
                "bearer_locked": 0,
                "last_ctx_needed": 0,
                "last_ctx_limit": 0,
                "last_ctx_ratio": 0.0,
            }
            _session_map[session_id] = s
            return s

        # Idle timeout downgrade: if the conversation is idle, revert to level 1.
        if (now - float(s.get("updated_at", 0.0))) > float(CAP_IDLE_TIMEOUT_S):
            s["capability_level"] = 1
            s["simple_streak"] = 0
            s["pinned_node"] = None
            s["pinned_model"] = None
            s["pinned_capability"] = None
            s["pinned_at"] = 0.0

        s["updated_at"] = now
        return s


def _session_pinned_tuple(s: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    pn = s.get("pinned_node")
    pm = s.get("pinned_model")
    if pn and pm:
        return (str(pn), str(pm))
    return None


def _session_controller_prepare(session: Dict[str, Any], session_id: str, lane: str) -> None:
    session["session_id"] = session_id
    session["lane"] = lane or session.get("lane") or "chat"
    session["updated_at"] = time.time()
    if not session.get("bearer_state"):
        session["bearer_state"] = "IDLE"

def _session_controller_decide(
    session: Dict[str, Any],
    *,
    lane: str,
    current_node: str,
    target_node: str,
    target_model: str,
    action: str,
    cause: str,
    ctx_needed: int,
    ctx_limit: int,
    ctx_ratio: float,
    prewarm: int,
    handover: int,
) -> None:
    session["lane"] = lane or session.get("lane") or "chat"
    session["current_action"] = action
    session["current_cause"] = cause or ""
    session["last_ctx_needed"] = int(ctx_needed or 0)
    session["last_ctx_limit"] = int(ctx_limit or 0)
    session["last_ctx_ratio"] = float(ctx_ratio or 0.0)
    if int(prewarm or 0) == 1:
        session["prewarmed_target"] = target_node
        session["prewarmed_model"] = target_model
        session["bearer_state"] = "PREWARMED"
    elif int(handover or 0) == 1:
        session["handover_pending"] = 1
        session["bearer_state"] = "HANDOVER_PENDING"
    else:
        session["handover_pending"] = 0
        session["bearer_state"] = "LOCKED" if int(session.get("bearer_locked") or 0) == 1 else "ACTIVE"

def _session_controller_commit_route(session: Dict[str, Any], *, node_name: str, model_id: str, stream: bool) -> None:
    session["current_node"] = node_name
    session["current_model"] = model_id
    session["bearer_locked"] = 1 if stream else 0
    session["bearer_state"] = "LOCKED" if stream else "ACTIVE"
    session["handover_pending"] = 0
    session["updated_at"] = time.time()

def _session_controller_complete(session: Dict[str, Any]) -> None:
    session["bearer_locked"] = 0
    session["bearer_state"] = "ACTIVE" if session.get("current_node") else "IDLE"
    session["updated_at"] = time.time()

def _session_controller_log(req_id: str, session: Dict[str, Any]) -> None:
    try:
        log.info(
            f"[{req_id}] → SCCI SESSION sid={session.get('session_id') or req_id} "
            f"lane={session.get('lane') or 'chat'} bearer_state={session.get('bearer_state') or 'IDLE'} "
            f"current={session.get('current_node') or '-'} model={session.get('current_model') or '-'} "
            f"prewarmed={session.get('prewarmed_target') or '-'} locked={int(session.get('bearer_locked') or 0)}"
        )
        log.info(
            f"[{req_id}] → SCCI BEARER_STATE sid={session.get('session_id') or req_id} "
            f"state={session.get('bearer_state') or 'IDLE'} ctx={int(session.get('last_ctx_needed') or 0)}/"
            f"{int(session.get('last_ctx_limit') or 0)} ratio={float(session.get('last_ctx_ratio') or 0.0):.2f}"
        )
    except Exception:
        pass


async def update_session_pin(session_id: str, s: Dict[str, Any], node: str, model: str) -> None:
    """Pin the session to a node/model at the *current* capability level."""
    async with _session_lock:
        s["pinned_node"] = node
        s["pinned_model"] = model
        s["pinned_capability"] = int(s.get("capability_level") or 1)
        s["pinned_at"] = time.time()
        s["updated_at"] = time.time()
        _session_map[session_id] = s


async def reset_session(session_id: str) -> None:
    """Manual / explicit reset: forget escalation and pinning (back to cap=1)."""
    async with _session_lock:
        _session_map[session_id] = {
            "capability_level": 1,
            "pinned_node": None,
            "pinned_model": None,
            "pinned_capability": None,
            "pinned_at": 0.0,
            "simple_streak": 0,
            "last_tech_ts": 0.0,
            "updated_at": time.time(),
        }


def user_explicit_reset(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for p in CAP_RESET_PHRASES:
        if p and p in t:
            return True
    # extra short manual reset command
    if t in ("/reset", "reset", "restart"):
        return True
    return False


# Technical signal (KISS heuristic):
# - code blocks / stacktraces are handled by CODE_PAT (already defined in v8)
# - deep reasoning keywords handled by DEEP_PAT (already defined in v8)
# - you can add keywords later by extending DEEP_PAT / CODE_PAT patterns in one place.
def is_technical_signal(text: str, bucket: str) -> bool:
    t = (text or "")
    if bucket in ("code", "deep", "vision_reasoning", "health"):
        return True
    if CODE_PAT.search(t) or DEEP_PAT.search(t):
        return True
    return False


def is_simple_turn(text: str) -> bool:
    """Return True when a message looks like 'small talk' / low-signal."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) > CAP_SIMPLE_MAX_CHARS:
        return False
    # If it includes obvious technical markers, it's not "simple".
    if CODE_PAT.search(t) or DEEP_PAT.search(t) or _is_medical_intent_text(t):
        return False
    return True


def capability_node_chain(cap_level: int, bucket: str, ctx_pressure_bn10: bool, need_vision: bool, need_gpu: bool) -> List[str]:
    # Tool bucket is a dedicated tier: XPZ (B-PHONE2) primary, PC-CPU fallback.
    if bucket == "tool":
        return ["B-PHONE2", "B-CPU0"]

    """Choose a deterministic node priority list for a session.

    Design goals (your lab philosophy):
    - Default chat stays on BN10 (phone) first, then PC-CPU, GPU last.
    - PC-CPU is the preferred "scale up" target when BN10 context is getting full.
    - GPU is used only when it is truly beneficial/required (vision, coding, etc.)
    - If a node is unhealthy/unready, routing will naturally fall through to the next.
    """
    cap = int(max(1, min(3, cap_level)))

    # Vision workloads: GPU-first, CPU fallback, BN10 last.
    if need_vision:
        return ["B-GPU0", "B-CPU0", "B-PHONE0"]

    if cap == 1:
        # Capacity fallback: when BN10 context is near full, go straight to PC-CPU same model.
        return ["B-CPU0", "B-GPU0", "B-PHONE0"] if ctx_pressure_bn10 else ["B-PHONE0", "B-CPU0", "B-GPU0"]

    if cap == 2:
        # GPU-first only when the request truly benefits (vision/code/health or explicit).
        return ["B-GPU0", "B-CPU0", "B-PHONE0"] if need_gpu else ["B-CPU0", "B-PHONE0", "B-GPU0"]

    # cap == 3: long context / complex => PC-CPU first (big ctx), then GPU, then BN10.
    return ["B-CPU0", "B-GPU0", "B-PHONE0"]

# ---------------- Bucket inference ----------------
def infer_bucket(req: Request, body: Dict[str, Any], text: str, img_count: int, prompt_tok_est: int, max_tokens: int) -> str:
    # Two-stage vision follow-up: if user asks for more details (no new image),
    # route to text-only reasoning using cached facts from last vision_fast.
    try:
        if img_count == 0 and is_detail_request(text or ""):
            sid = body.get("chat_id") or req.headers.get("x-openwebui-chat-id") or body.get("session") or ""
            if sid and get_cached_vision_facts(str(sid)):
                return "vision_detail_text"
    except Exception:
        pass

    # Tool-selection meta prompts (e.g., OpenWebUI tool router).
    # NOTE: OpenWebUI places the tool-selection instruction in the SYSTEM message,
    # so we scan all message contents (not only last_user_text).
    try:
        msgs = body.get("messages") or []
        for m in msgs:
            c = m.get("content", "")
            # content can be str or list (multimodal); stringify safely
            if TOOL_SELECT_PAT.search(str(c)):
                return "tool"
    except Exception:
        pass

    hb = (req.headers.get("x-bucket") or "").strip().lower()
    if hb:
        return hb
    rb = (body.get("router_bucket") or "").strip().lower()
    if rb:
        return rb

    is_ops_health = bool(OPS_HEALTH_PAT.search(text))
    medical_intent = _is_medical_intent_text(text)

    if img_count > 0:
        if CODE_PAT.search(text):
            return "code"
        if (not is_ops_health) and medical_intent:
            return "health"
        if VISION_REASON_PAT.search(text):
            return "vision_reasoning"
        return "vision_fast"

    if CODE_PAT.search(text):
        return "code"
    if (not is_ops_health) and medical_intent:
        return "health"
    if MICRO_PAT.search(text):
        return "micro"
    if DEEP_PAT.search(text):
        return "deep"

    if (prompt_tok_est + max_tokens) > int(node_max_ctx("B-PHONE0") * 0.75):
        return "longctx"

    return "chat"

def bucket_node_order(bucket: str) -> List[str]:
    return DEFAULT_BUCKET_NODE_ORDER.get(bucket) or DEFAULT_BUCKET_NODE_ORDER["chat"]

# ---------------- Model selection ----------------
def find_model_by_hint(node: Node, hint: str, bucket: str = "chat") -> Optional[str]:
    if not hint:
        return None
    hl = hint.lower()
    for mid in node.models.keys():
        # Function/tool models must not serve non-tool buckets
        if ("bucket" in locals() and bucket != "tool") and FUNCTION_MODEL_PAT.search(str(mid)):
            continue
        if mid.lower() == hl:
            return mid
    for mid in node.models.keys():
        # Function/tool models must not serve non-tool buckets
        if ("bucket" in locals() and bucket != "tool") and FUNCTION_MODEL_PAT.search(str(mid)):
            continue
        if hl in mid.lower():
            return mid
    return None

def pick_best_for_bucket(node: Node, bucket: str, need_vision: bool) -> Optional[str]:
    if not node.models:
        return None
    candidates: List[Tuple[int, str]] = []
    for m in node.models.values():
        if need_vision and not m.vision:
            continue
        if (not need_vision) and m.vision and bucket not in ("vision_fast", "vision_reasoning", "health"):
            continue
        candidates.append((int(m.bucket_scores.get(bucket, 0)), m.id))
    if not candidates:
        for m in node.models.values():
            if need_vision and m.vision:
                candidates.append((0, m.id))
            elif (not need_vision) and (not m.vision):
                candidates.append((0, m.id))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def ctx_fits(node_name: str, prompt_tok_est: int, max_tokens: int) -> bool:
    budget = int(node_max_ctx(node_name) * CTX_HEADROOM_RATIO)
    return (prompt_tok_est + max_tokens) <= budget

def should_avoid_switch(node: Node) -> bool:
    return node.name in ROUTER_MODE_NODES

def get_sticky_model(node: Node, bucket: str) -> Optional[str]:
    v = node.last_model_by_bucket.get(bucket)
    if not v:
        return None
    mid, ts = v
    if (time.time() - ts) <= NODE_MODEL_STICKY_S:
        return mid
    return None

def set_sticky_model(node: Node, bucket: str, model: str) -> None:
    node.last_model_by_bucket[bucket] = (model, time.time())

def bucket_switch_allowed(node: Node, new_bucket: str) -> bool:
    prev_bucket, ts = node.last_bucket
    if not prev_bucket or prev_bucket == new_bucket:
        return True
    return (time.time() - ts) >= NODE_SWITCH_COOLDOWN_S

def record_bucket(node: Node, bucket: str) -> None:
    node.last_bucket = (bucket, time.time())

# ---------------- Routing scoring ----------------
def _norm_latency_ms(ms: float) -> float:
    ms = max(50.0, min(ms, 20000.0))
    return math.log(ms / 50.0 + 1.0)

def _norm_load(x: int) -> float:
    x = max(0, min(x, 100))
    return math.log(x + 1.0)

def _norm_queue(x: int) -> float:
    x = max(0, min(x, 200))
    return math.log(x + 1.0)

def score_node_for_request(node: Node, bucket: str, sticky_hit: bool, switch_allowed: bool) -> float:
    lat_ms = node.lat_ema_by_bucket.get(bucket, node.lat_ema_ms)
    s = 0.0
    s += W_LAT * _norm_latency_ms(lat_ms)

    if node.metrics_fresh():
        s += W_LOAD * _norm_load(node.metrics.active_requests)
        s += W_QUEUE * _norm_queue(node.metrics.queue_depth)
        if node.metrics.loading:
            s += 10.0
    else:
        s += 0.5

    if sticky_hit:
        s -= W_STICKY
    if not switch_allowed:
        s += W_SWITCH_PENALTY
    if should_avoid_switch(node) and not sticky_hit:
        s += W_ROUTERMODE_PENALTY
    if bucket in node.tags:
        s -= 0.1
    return s

# ---------------- Identity helpers ----------------
def extract_user_id(req: Request, body: Dict[str, Any]) -> Optional[str]:
    for k in ("x-user-id", "x-openwebui-user-id"):
        v = (req.headers.get(k) or "").strip()
        if v:
            return v
    u = body.get("user")
    if u:
        return str(u)
    return None

def extract_chat_id(req: Request, body: Dict[str, Any]) -> Optional[str]:
    for k in ("x-chat-id", "x-conversation-id", "x-openwebui-chat-id"):
        v = (req.headers.get(k) or "").strip()
        if v:
            return v
    for key in ("conversation_id", "chat_id", "thread_id"):
        v = body.get(key)
        if v:
            return str(v)
    return None

# ---------------- Gatekeeper runner (mem_gate) ----------------
async def run_gatekeeper_mem_gate(user_text: str) -> Optional[int]:
    forced = gatekeeper_prefilter_state(user_text)
    if forced is not None:
        log.info(f"[mem_gate] prefilter forced_state={forced} head={user_text[:80]!r}")
        return forced

    async with GATEKEEPER_SEMAPHORE:
        # Allowed states come from policy (defaults to [0,1,2,3])
        allowed = gatekeeper_allowed_states()
        payload = {
            "model": GATEKEEPER_MODEL_ID,
            "stream": False,
            "temperature": 0,
            "max_tokens": GATEKEEPER_MAX_TOKENS,
            "router_bucket": "mem_gate",
            "messages": [{"role": "user", "content": gatekeeper_prompt(user_text)}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gatekeeper_1b",
                    "schema": {
                        "type": "object",
                        "properties": {"state": {"type": "integer", "enum": allowed}},
                        "required": ["state"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        order = DEFAULT_BUCKET_NODE_ORDER.get("mem_gate", ["B-PHONE1", "B-CPU0"])
        tried: List[str] = []

        for node_name in order:
            node = _nodes.get(node_name)
            if not node:
                continue
            try:
                ok = node.healthy_cached() or await check_node_health(node)
                if not ok:
                    tried.append(f"{node_name}(unhealthy)")
                    continue
                if not await check_node_ready(node):
                    tried.append(f"{node_name}(not_ready)")
                    continue

                await discover_models_for_node(node)

                model_id = None
                if GATEKEEPER_MODEL_ID in node.models:
                    model_id = GATEKEEPER_MODEL_ID
                else:
                    hint = bucket_hint(node_name, "mem_gate") or GATEKEEPER_MODEL_ID
                    model_id = find_model_by_hint(node, hint, bucket)

                if not model_id:
                    tried.append(f"{node_name}(no_gate_model)")
                    continue

                payload2 = dict(payload)
                payload2["model"] = model_id

                r = await client.post(node.url(OPENAI_CHAT_PATH), headers={}, json=payload2)
                if r.status_code >= 400:
                    tried.append(f"{node_name}(http_{r.status_code})")
                    continue

                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                st = parse_gatekeeper_state(content, allowed=allowed)
                if ROUTER_LOG_MEMORY_WRITES:
                    head = (content or "")[:240].replace("\n", " ")
                    log.info(f"[mem_gate] node={node_name} model={model_id} raw_head={head!r} parsed={st}")
                tried.append(f"{node_name}(ok:{st})")
                if st in allowed:
                    return st
                tried.append(f"{node_name}(parse_fail)")

            except Exception as e:
                tried.append(f"{node_name}(exc:{type(e).__name__})")
                continue

        log.info(f"[mem_gate] failed tried={tried}") if ROUTER_LOG_MEMORY_WRITES else log.debug(f"[mem_gate] failed tried={tried}")
        return None



async def memory_write_task(user_text: str, session_id: str, user_id: Optional[str], chat_id: Optional[str]) -> None:
    try:
        clean = sanitize_for_memory_save(user_text)
        if not clean:
            return
        # Belt+suspenders: never store OpenWebUI meta-tasks
        if is_openwebui_task_prompt(clean):
            return

        # v9.1.1 strict prefilter: avoid saving generic Q/A or assistant-style trivia.
        if not is_memory_save_candidate(clean):
            return

        # Memory gate: default to LLM gatekeeper (BN8/PC-CPU) for correctness.
        # Optional quick heuristic classifier can be enabled to reduce latency.
        st = None
        if ROUTER_QUICK_MEMORY_CLASSIFIER:
            st = quick_memory_state(clean)

        if st is None:
            st = await run_gatekeeper_mem_gate(clean)

        if ROUTER_LOG_MEMORY_WRITES:
            log.info(f"[memory_gate] state={st} text={clean[:120]!r}")
        # Policy-driven behavior (KISS default, scalable later):
        # - By default (no policy override), this behaves exactly like v8.1.4:
        #     state 1 -> save to Profile
        #     state 2 -> save to Project
        #     others  -> ignore
        #
        # - To add more states later (3,4,5...), set one of:
        #     ROUTER_GATEKEEPER_POLICY_JSON   or   ROUTER_GATEKEEPER_POLICY_PATH
        #   Example for 5 states (JSON):
        #     {"version":1,"states":{
        #       "0":{"action":"ignore"},
        #       "1":{"action":"save","scope":"profile","kind":"profile"},
        #       "2":{"action":"save","scope":"project","kind":"project"},
        #       "3":{"action":"save","scope":"project","kind":"task","importance":60},
        #       "4":{"action":"save","scope":"profile","kind":"preference","importance":55}
        #     }}
        rule = gatekeeper_rule_for_state(st)
        if not rule:
            # Most common cause: policy override with unexpected key types (e.g. "1" vs 1).
            log.warning(f"[memory_gate] no_rule_for_state={st} (check ROUTER_GATEKEEPER_POLICY_JSON/PATH). Ignoring.")
            return
        if (rule.get("action") or "").lower() != "save":
            # Explicit ignore in policy.
            log.info(f"[memory_gate] policy_ignore state={st} name={rule.get('name')!r}")
            return

        # v8.1.4 hybrid scope resolver:
        # If a chat_id exists, treat it as an "active project" session => save to PROJECT.
        # Otherwise save to PROFILE.
        scope = SCOPE_PROJECT if session_has_active_project(chat_id) else SCOPE_PROFILE
        # Allow policy override; else keep your heuristic.
        kind = (rule.get("kind") or "").strip() or _guess_kind_from_text(clean)

        # Allow per-state importance override; else use default.
        imp = rule.get("importance")
        if imp is None:
            imp = MEMORY_WRITE_DEFAULT_IMPORTANCE
        imp = int(max(0, min(int(imp), 100)))

        if ROUTER_DB_WRITE_ASYNC:
            _enqueue_memory_write(scope, clean, session_id, user_id, chat_id, kind, imp, 100)
        else:
            await asyncio.to_thread(
                db_insert_memory_v8, scope, clean, session_id, user_id, chat_id, kind, imp, 100
            )
        if ROUTER_LOG_MEMORY_WRITES:
            log.info(f"[memory_write] v8 scope={scope} kind={kind} imp={imp} session={session_id} user={user_id} chat={chat_id} text={clean[:120]!r}")
    except Exception as e:
        log.warning(f"[memory_write] failed: {e!r}", exc_info=True)

async def memory_read_v8(
    scope: int,
    user_id: Optional[str],
    chat_id: Optional[str],
    session_id: str,
    since_ts: Optional[int],
    top_k: int
) -> Tuple[List[Dict[str, Any]], List[int]]:
    # fetch candidates then score them (fast enough for small LAN)
    cand = await asyncio.to_thread(
        db_fetch_candidates_v8, scope, user_id, chat_id, session_id, since_ts, MEMORY_CANDIDATE_LIMIT
    )
    if not cand:
        return [], []

    now_ts = int(time.time())
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in cand:
        try:
            scored.append((_score_memory_row(now_ts, r), r))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [r for _, r in scored[:top_k]]
    ids = [int(r["id"]) for r in picked if "id" in r]
    return picked, ids

# ---------------- FastAPI app ----------------
log.info("[DEMO_MODE] enabled=%s discovery_interval_s=%s changes_only=%s", ROUTER_DEMO_MODE, ROUTER_DISCOVERY_INTERVAL_S, ROUTER_DISCOVERY_LOG_CHANGES_ONLY)
log.info("[TELECOM_CLEAN] enabled=%s request_logs=%s header_logs=%s verbose_headers=%s", ROUTER_TELECOM_CLEAN_MODE, ROUTER_LOG_REQUESTS, ROUTER_LOG_HEADERS, ROUTER_LOG_HEADERS_VERBOSE)

app = FastAPI(title="Lab LLM Router (capability+sticky+fallback+metrics+state+handoverfix)", version="10.1.3")

# Expose generated images so returned PNG URLs resolve in OpenWebUI/browser.
app.mount(
    "/generated_images",
    StaticFiles(directory=str(_image_gen_output_dir())),
    name="generated_images",
)

@app.middleware("http")
async def router_debug_header_middleware(request: Request, call_next):
    response = await call_next(request)
    if ROUTER_DEBUG_HEADERS:
        try:
            for k, v in (
                ("x-router-bucket", getattr(request.state, "router_bucket", None)),
                ("x-router-node", getattr(request.state, "router_node", None)),
                ("x-router-model", getattr(request.state, "router_model", None)),
                ("x-router-reason", getattr(request.state, "router_reason", None)),
                ("x-router-capability-level", getattr(request.state, "router_capability_level", None)),
                ("x-router-pinned", getattr(request.state, "router_pinned", None)),
            ):
                if v is not None:
                    response.headers[k] = str(v)
        except Exception:
            pass
    return response

@app.middleware("http")
async def router_observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    await ROUTER_STATE.set_async(
        "RECEIVED",
        event="request_received",
        path=request.url.path,
        method=request.method,
    )
    if ROUTER_OBS_ENABLE:
        await ROUTER_OBS.request_started()
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        bucket = response.headers.get("X-Bucket") or getattr(request.state, "router_bucket", "")
        node = response.headers.get("X-Node") or getattr(request.state, "router_node", "")
        model = response.headers.get("X-Model") or getattr(request.state, "router_model", "")
        state_name = "COMPLETED"
        event_name = "response_completed"
        if str(node or "").upper() == "B-CPU0":
            state_name = "HANDOVER"
            event_name = "cpu_handover"
        elif str(node or "").upper() == "B-GPU0":
            state_name = "HANDOVER"
            event_name = "gpu_handover"
        elif str(node or "").upper() in {"B-PHONE0", "B-PHONE3"}:
            state_name = "ROUTED"
            event_name = "edge_route"
        await ROUTER_STATE.set_async(
            state_name,
            node=node,
            bucket=bucket,
            model=model,
            event=event_name,
            path=request.url.path,
            method=request.method,
        )
        if ROUTER_OBS_ENABLE:
            await ROUTER_OBS.request_finished(
                path=request.url.path,
                status_code=getattr(response, "status_code", 200),
                latency_ms=latency_ms,
                bucket=bucket,
                node=node,
                model=model,
            )
        return response
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        await ROUTER_STATE.set_async(
            "ERROR",
            event="request_failed",
            path=request.url.path,
            method=request.method,
        )
        if ROUTER_OBS_ENABLE:
            await ROUTER_OBS.request_failed(path=request.url.path, latency_ms=latency_ms)
        raise

@app.on_event("startup")
async def startup():
    db_init()

    # v8.1.6 perf: start background memory write workers (optional)
    # - Default ON (ROUTER_DB_WRITE_ASYNC=1)
    # - Queue is bounded; if full we drop writes (best-effort memory, protects latency)
    global _MEMORY_WRITE_QUEUE, _MEMORY_WRITE_TASKS
    _MEMORY_WRITE_QUEUE = asyncio.Queue(maxsize=max(1, ROUTER_DB_WRITE_QUEUE_MAX))
    _MEMORY_WRITE_TASKS = []
    if ROUTER_DB_WRITE_ASYNC:
        nworkers = max(1, ROUTER_DB_WRITE_WORKERS)
        # SQLite: multiple writers can increase lock contention; keep 1 unless you measured otherwise.
        _MEMORY_WRITE_TASKS = [asyncio.create_task(_memory_write_worker(i + 1)) for i in range(nworkers)]

    _nodes.clear()
    for name, url in DEFAULT_NODES.items():
        if not url:
            continue
        tags = list(DEFAULT_NODE_TAGS.get(name, []))
        _nodes[name] = Node(name=name, base_url=url, tags=tags)

    asyncio.create_task(discovery_loop())

    log.info("=================================================")
    log.info("SCCI LAB ROUTER v9.2.0 INITIALIZED (StructuredMemory+Decay+ConditionalTime+ReqLogging+GateDebug+ImageGenNativeLane)")
    log.info("-------------------------------------------------")
    log.info("Architecte & Author: Benslaiman.com Contact Email: contact@benslaiman.com")
    log.info("AI-assisted desing & Implementation (used as cognitive accelerator)")
    log.info("SCCI (Smart Cognitive Cluster Intelligence) | Telecom Legacy Log Mode | Session Controller v12.4 Image Diagram Route + Image Generation")
    log.info(f"DB: {str(DB_PATH.resolve())}")
    log.info(f"TZ: {TZ_NAME or 'system'}")
    log.info(f"Nodes: {', '.join([f'{n}={_nodes[n].base_url}' for n in _nodes])}")
    log.info(f"Discovery interval: {DISCOVERY_INTERVAL_S}s")
    log.info(f"Router-mode nodes: {', '.join(ROUTER_MODE_NODES) if ROUTER_MODE_NODES else '(none)'}")
    log.info(f"TOOLS lane: FunctionGemma -> {FUNCTION_TOOL_NODE_PRIMARY} model={FUNCTION_TOOL_MODEL_ID} fallback={FUNCTION_TOOL_NODE_FALLBACK}")
    log.info(f"UI helper lane: {UI_HELPER_NODE_PRIMARY} model={UI_HELPER_MODEL_ID} fallback={UI_HELPER_NODE_FALLBACK}")
    log.info(f"Datetime tool demo: enabled={int(DATETIME_TOOL_ENABLE or 0)} cities={TIME_TOOL_CITIES}")
    log.info(f"LongText history: keep_last_n={LONGTEXT_HISTORY_MESSAGES} char_budget={LONGTEXT_HISTORY_CHAR_BUDGET}")
    log.info("LongText history alignment: first conversational role forced=user, alternation enforced")
    log.info("LongText history mode: recent chat compressed into SYSTEM context block + final USER document")
    log.info(f"Tools trim: enabled={TOOLS_TRIM_ENABLED} max_chars={TOOLS_TRIM_MAX_CHARS} chat_history_max_chars={TOOLS_CHAT_HISTORY_MAX_CHARS}")
    log.info(f"Vision text intent: enabled={ROUTER_VISION_TEXT_INTENT}")
    log.info("Vision safeguard scope: current_turn_images_only=1")
    log.info(f"LongText escalation: buffer_ratio={LONGTEXT_ESCALATION_BUFFER_RATIO} emergency_tokens={LONGTEXT_EMERGENCY_TOKENS} ladder={CHAT_LADDER_MODELS}")
    log.info(f"LongText admission: direct_threshold={LONGTEXT_DIRECT_THRESHOLD} color_logs={1 if ROUTER_COLOR_LOGS else 0} emergency_restore=1 stepup_handover=1")
    log.info("Vision safeguard: preserve original multimodal payload and bypass longtext rewrite on image requests")
    log.info(f"Vision preprocess: enabled={int(ROUTER_VISION_PREPROCESS_ENABLE or 0)} size={int(ROUTER_VISION_PREPROCESS_SIZE or 896)} bg={ROUTER_VISION_PREPROCESS_BG} pillow={1 if Image is not None else 0}")
    log.info(f"ReqLog: {ROUTER_LOG_REQUESTS} (0=off,1=last_user,2=roles,3=full)")
    log.info(f"MemoryV8: {1 if MEMORY_ENABLE_V8 else 0} half_life_days={MEMORY_DECAY_HALFLIFE_DAYS} recall_boost={MEMORY_RECALL_BOOST}")
    log.info(f"DBWriteAsync: {1 if ROUTER_DB_WRITE_ASYNC else 0} queue_max={ROUTER_DB_WRITE_QUEUE_MAX} workers={ROUTER_DB_WRITE_WORKERS}")
    try:
        log.info(f"GatekeeperPolicy states={gatekeeper_allowed_states()} model={GATEKEEPER_MODEL}")
    except Exception:
        pass
    log.info(f"Procedures: PREWARM={ROUTER_PREWARM_ENABLED} HANDOVER=1 BEARER_LOCK=1 prewarm_ratio={ROUTER_CTX_PREWARM_RATIO} handover_ratio={ROUTER_CTX_HANDOVER_RATIO} reserve={ROUTER_CTX_COMPLETION_RESERVE} cooldown_s={ROUTER_PREWARM_COOLDOWN_S}")
    log.info(f"Observability: enabled={1 if ROUTER_OBS_ENABLE else 0} log_buffer_max={ROUTER_OBS_LOG_BUFFER_MAX} sse_heartbeat_s={ROUTER_OBS_SSE_HEARTBEAT_S}")
    log.info(f"Chat ladder nodes: {CHAT_LADDER_NODES}")
    log.info(f"Chat failover order: {ROUTER_CHAT_FAILOVER_ORDER}")
    log.info(f"Image generation lane order: {IMAGE_GEN_BACKEND_ORDER}")
    log.info(f"Prewarm phone wake nodes: {sorted(list(ROUTER_PREWARM_PHONE_WAKE_NODES))} retry_attempts={ROUTER_PREWARM_PHONE_RETRY_ATTEMPTS} retry_delays_s={ROUTER_PREWARM_PHONE_RETRY_DELAYS_S}")
    log.info("Endpoints: /health /v1/models /v1/chat/completions /router/metrics /router/state /router/topology /router/logs")
    log.info("=================================================")

@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
    await stream_client.aclose()

@app.get("/health")
async def health():
    nodes = []
    for n in _nodes.values():
        nodes.append({
            "name": n.name,
            "base_url": n.base_url,
            "tags": n.tags,
            "status": n.status,
            "ready": n.ready,
            "ready_status": n.ready_status,
            "ready_supported": n.ready_supported,
            "metrics_supported": n.metrics_supported,
            "healthy_cached": n.healthy_cached(),
            "models": len(n.models),
            "max_ctx": node_max_ctx(n.name),
            "lat_ema_ms": n.lat_ema_ms,
        })
    return {
        "ok": True,
        "db": str(DB_PATH.resolve()),
        "tz": (TZ_NAME or "system"),
        "memory_v8": bool(MEMORY_ENABLE_V8),
        "nodes": nodes
    }

@app.get("/router/metrics")
async def router_metrics():
    out = {}
    for name, n in _nodes.items():
        out[name] = {
            "base_url": n.base_url,
            "status": n.status,
            "ready": n.ready,
            "ready_status": n.ready_status,
            "ready_supported": n.ready_supported,
            "metrics_supported": n.metrics_supported,
            "fail_streak": n.fail_streak,
            "lat_ema_ms": n.lat_ema_ms,
            "lat_ema_by_bucket": n.lat_ema_by_bucket,
            "metrics": {
                "active_requests": n.metrics.active_requests,
                "queue_depth": n.metrics.queue_depth,
                "loading": n.metrics.loading,
                "tokens_per_s": n.metrics.tokens_per_s,
                "last_updated": n.metrics.last_updated,
            },
            "last_model_by_bucket": n.last_model_by_bucket,
            "last_bucket": n.last_bucket,
        }
    obs = await ROUTER_OBS.snapshot() if ROUTER_OBS_ENABLE else {}
    return {
        "ok": True,
        "db": str(DB_PATH.resolve()),
        "tz": (TZ_NAME or "system"),
        "router": out,
        "observability": obs,
        "summary": {
            "nodes_total": len(_nodes),
            "nodes_ok": sum(1 for n in _nodes.values() if str(getattr(n, "status", "")) == "ok"),
            "nodes_ready": sum(1 for n in _nodes.values() if bool(getattr(n, "ready", False))),
            "requests_total": obs.get("requests_total", 0),
            "requests_in_flight": obs.get("requests_in_flight", 0),
            "avg_latency_ms": obs.get("avg_latency_ms", 0),
            "last_latency_ms": obs.get("last_latency_ms", 0),
            "sse_clients": obs.get("sse_clients", 0),
        },
    }

@app.get("/router/state")
async def router_state():
    state = await ROUTER_STATE.snapshot()
    return {
        "ok": True,
        "router": {
            "name": "SCCI-Router",
            "version": app.version,
        },
        "state": state,
    }

@app.get("/router/topology")
async def router_topology():
    nodes = []
    for name, n in _nodes.items():
        nodes.append({
            "name": name,
            "base_url": n.base_url,
            "tags": list(getattr(n, "tags", []) or []),
            "status": n.status,
            "ready": n.ready,
            "ready_status": n.ready_status,
            "models": sorted(list(n.models.keys())),
            "max_ctx": node_max_ctx(name),
            "metrics": {
                "active_requests": n.metrics.active_requests,
                "queue_depth": n.metrics.queue_depth,
                "loading": n.metrics.loading,
                "tokens_per_s": n.metrics.tokens_per_s,
                "last_updated": n.metrics.last_updated,
            },
        })

    return {
        "ok": True,
        "router": {
            "name": "SCCI-Router",
            "version": app.version,
            "started_at": ROUTER_STARTED_AT,
        },
        "nodes": nodes,
        "links": [
            {"from": "clients", "to": "SCCI-Router", "type": "ingress"},
            *[{"from": "SCCI-Router", "to": n["name"], "type": "inference"} for n in nodes],
        ],
    }

@app.get("/router/logs")
async def router_logs(request: Request, since: int = 0):
    async def event_iter():
        last_seq = max(0, int(since or 0))
        heartbeat_s = max(1, ROUTER_OBS_SSE_HEARTBEAT_S)
        last_emit = time.time()
        if ROUTER_OBS_ENABLE:
            async with ROUTER_OBS.lock:
                ROUTER_OBS.sse_clients += 1
        try:
            while True:
                if await request.is_disconnected():
                    break

                batch = []
                if ROUTER_OBS_ENABLE:
                    async with ROUTER_OBS.lock:
                        for item in ROUTER_OBS.log_buffer:
                            if int(item.get("seq", 0)) > last_seq:
                                batch.append(dict(item))

                if batch:
                    for item in batch:
                        last_seq = int(item.get("seq", last_seq))
                        payload = json.dumps(item, ensure_ascii=False)
                        yield f"id: {last_seq}\nevent: log\ndata: {payload}\n\n"
                    last_emit = time.time()
                else:
                    if (time.time() - last_emit) >= heartbeat_s:
                        yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time()}, ensure_ascii=False)}\n\n"
                        last_emit = time.time()
                    await asyncio.sleep(0.5)
        finally:
            if ROUTER_OBS_ENABLE:
                async with ROUTER_OBS.lock:
                    ROUTER_OBS.sse_clients = max(0, ROUTER_OBS.sse_clients - 1)

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/v1/models")
async def models():
    # KEEP SCCI-Router + tagged models
    data = [{"id": "SCCI-Router", "object": "model"}]
    for node in _nodes.values():
        for mid in sorted(node.models.keys()):
            # Function/tool models must not serve non-tool buckets
            if ("bucket" in locals() and bucket != "tool") and FUNCTION_MODEL_PAT.search(str(mid)):
                continue
            data.append({"id": f"[{node.name}] {mid}", "object": "model"})
    return {"object": "list", "data": data}

def set_headers(
    resp,
    req_id: str,
    session_id: str,
    bucket: str,
    node: str,
    model: str,
    reason: str,
    tried: List[str],
    extra: Optional[Dict[str, str]] = None,
):
    resp.headers["X-Req-Id"] = req_id
    resp.headers["X-Session-Id"] = session_id
    resp.headers["X-Bucket"] = bucket
    resp.headers["X-Node"] = node
    resp.headers["X-Model"] = model
    resp.headers["X-Reason"] = reason
    resp.headers["X-Tried"] = " | ".join(tried)
    if extra:
        for k, v in extra.items():
            resp.headers[k] = v


def _normalize_node_key(name: str) -> Optional[str]:
    """Resolve node name case-insensitively to keys in _nodes."""
    if not name:
        return None
    n = name.strip()
    if not n:
        return None
    if n in _nodes:
        return n
    nl = n.lower()
    if nl in _nodes:
        return nl
    for k in _nodes.keys():
        if k.lower() == nl:
            return k
    return None

def parse_direct_model(model_str: str) -> Tuple[Optional[str], str]:
    """Parse model strings like '[node] model' (space optional)."""
    if model_str is None:
        return None, ""
    s = str(model_str).strip()
    if not s:
        return None, s
    m = re.match(r"^\[(?P<node>[^\]]+)\]\s*(?P<model>.*)$", s)
    if m:
        node = (m.group("node") or "").strip()
        mdl = (m.group("model") or "").strip()
        return node, mdl
    return None, s


# ---------------- Chat endpoint ----------------
@app.post("/v1/chat/completions")
async def chat(req: Request):

    # v32: initialize routing variables (clean architecture guard)
    bucket = None
    # v32.4: bucket must never be None (OpenWebUI tool pre-pass / edge cases)
    if bucket is None:
        bucket = "chat"
    node_name = None
    model_id = None
    cap_level = 0
    reason = ""
    req_id = make_req_id()
    started = time.time()

    tried: List[str] = []  # v33.7.5: track attempted nodes for fallback/metrics
    _debug_log_headers(req)

    body = await req.json()

    # v33: always define stream early
    stream = bool(body.get("stream", False))

    # v33.7.8: vision defaults to avoid UnboundLocalError
    img_count = 0
    need_vis = False
    # v12.0: SCCI System Knowledge direct control-plane responses
    try:
        _sys_qa_last_user = _last_user_text(body.get("messages") or []) if isinstance(body, dict) else ""
        _sys_qa = _scci_system_knowledge_match(_sys_qa_last_user)
        if _sys_qa:
            if SCCI_SYSTEM_KNOWLEDGE_LOG:
                log.info(f"[{req_id}] → SCCI SYS_KNOWLEDGE sid={req_id} intent={_sys_qa['intent']} direct=1")
                log.info(f"[{req_id}] → SCCI DIRECT_RESPONSE sid={req_id} source=system_knowledge intent={_sys_qa['intent']}")
            if stream:
                return _scci_direct_streaming_response(req_id, _sys_qa["answer"])
            await _qa_natural_delay(_sys_qa["answer"])
            return JSONResponse(content=_build_direct_chat_response(req_id, _sys_qa["answer"]))
    except Exception as _sys_qa_err:
        log.warning(f"[{req_id}] scci_system_knowledge_err={type(_sys_qa_err).__name__}:{_sys_qa_err}")

    try:
        _demo_prompt = _current_turn_demo_text(body.get("messages") or []) if isinstance(body, dict) else ""
        _time_det = _detect_datetime_tool_request(_demo_prompt)
        if _time_det:
            _validated, _time_src = (False, "deterministic")
            if _time_det.get("mode") == "current" and int(DATETIME_TOOL_VALIDATE_SIMPLE or 0):
                _validated, _time_src = await _time_tool_functiongemma_validate(req_id, _time_det)
            _time_answer = _format_datetime_tool_answer(_time_det)
            log.info(
                f"[{req_id}] → SCCI DIRECT_RESPONSE sid={req_id} source=datetime_tool mode={_time_det.get('mode') or 'current'} city={_time_det.get('city') or _time_det.get('target_city') or '-'} "
                f"kind={_time_det.get('kind') or 'time'} validated={1 if _validated else 0} via={_time_src}"
            )
            if stream:
                return _scci_direct_streaming_response(req_id, _time_answer)
            await _qa_natural_delay(_time_answer)
            return JSONResponse(content=_build_direct_chat_response(req_id, _time_answer))

        _ops_det = _detect_scci_ops_request(_demo_prompt)
        if _ops_det:
            _ops_answer = _build_scci_ops_answer(_ops_det)
            log.info(
                f"[{req_id}] → SCCI DIRECT_RESPONSE sid={req_id} source=scci_ops kind={_ops_det['kind']} node={_ops_det.get('node') or '-'}"
            )
            if stream:
                return _scci_direct_streaming_response(req_id, _ops_answer)
            await _qa_natural_delay(_ops_answer)
            return JSONResponse(content=_build_direct_chat_response(req_id, _ops_answer))

        if _detect_demo_failover_control_intent(_demo_prompt):
            _failover_status = await _check_image_failover_readiness()
            log.info(
                f"[{req_id}] → SCCI DEMO_FAILOVER_CONTROL sid={req_id} primary={_failover_status['primary_node']} fallback={_failover_status['fallback_node']} direct=1"
            )
            log.info(
                f"[{req_id}] → SCCI FAILOVER_READINESS sid={req_id} primary={'UP' if _failover_status['primary_up'] else 'DOWN'} fallback={'UP' if _failover_status['fallback_up'] else 'DOWN'}"
            )
            _failover_answer = _build_demo_failover_control_answer(_failover_status)
            log.info(f"[{req_id}] → SCCI DIRECT_RESPONSE sid={req_id} source=demo_failover_control")
            if stream:
                return _scci_direct_streaming_response(req_id, _failover_answer)
            await _qa_natural_delay(_failover_answer)
            return JSONResponse(content=_build_direct_chat_response(req_id, _failover_answer))

        if _detect_architecture_diagram_intent(_demo_prompt):
            _diagram_path = _architecture_diagram_path()
            if not _diagram_path.exists():
                raise FileNotFoundError(str(_diagram_path))
            _diagram_url = _architecture_diagram_public_url()
            log.info(f"[{req_id}] SCCI INTENT=architecture_diagram")
            log.info(f"[{req_id}] SCCI DIAGRAM_OK asset={ARCH_DIAGRAM_FILENAME} path={_diagram_path}")
            _diagram_content = _build_architecture_markdown_reply(_diagram_url)
            if stream:
                return _scci_direct_streaming_response(req_id, _diagram_content)
            return JSONResponse(content=_build_direct_chat_response(req_id, _diagram_content))

        if _detect_image_generation_intent(_demo_prompt):
            log.info(f"[{req_id}] SCCI INTENT=image_generation")
            _img_resp = await _call_image_generation_backend(req_id, _demo_prompt)
            _images = _img_resp.get("images") or []
            if not _images:
                raise RuntimeError("image_generation returned no images")
            _saved_path, _public_url = _save_generated_image(req_id, _images[0])
            _backend_used = str(_img_resp.get("backend") or "")
            _node_used = str(_img_resp.get("node") or _backend_used or "image-engine")
            log.info(f"[{req_id}] SCCI IMAGE_GEN_OK backend={_backend_used} node={_node_used} saved={_saved_path}")
            _img_content = _build_image_markdown_reply(_public_url, _node_used)
            if stream:
                return _scci_direct_streaming_response(req_id, _img_content)
            return JSONResponse(content=_build_direct_chat_response(req_id, _img_content))
    except FileNotFoundError as _diagram_err:
        log.error(f"[{req_id}] SCCI DIAGRAM_FAIL err={type(_diagram_err).__name__}:{_diagram_err}")
        _diagram_fail = f"I could not find the LAB Router architecture diagram. Please place {ARCH_DIAGRAM_FILENAME} in {IMAGE_GEN_OUTPUT_DIR}."
        if stream:
            return _scci_direct_streaming_response(req_id, _diagram_fail)
        return JSONResponse(content=_build_direct_chat_response(req_id, _diagram_fail))
    except Exception as _time_tool_err:
        log.warning(f"[{req_id}] scci_time_tool_err={type(_time_tool_err).__name__}:{_time_tool_err}")

    # v30: If OpenWebUI sends a tool pre-pass for a huge summarize request, bypass tooling entirely (avoid B-PHONE2)
    if ROUTER_LONGPASTE_ENABLE:
        try:
            _msgs = body.get("messages", []) if isinstance(body, dict) else []
            _utxt = _last_user_text(_msgs)
            if _looks_like_summarize_request(_msgs) and len(_utxt) >= ROUTER_LONGPASTE_CHAR_THRESHOLD:
                # Detect tool pre-pass via system prompt header
                if isinstance(_msgs, list) and len(_msgs) > 0 and (str(_msgs[0].get("role","")) == "system"):
                    sc = _msgs[0].get("content","")
                    if isinstance(sc, str) and "Available Tools" in sc:
                        if ROUTER_DEBUG_HEADERS:
                            try:
                                req.state.router_bucket = "tool"
                                req.state.router_node = "bypass"
                                req.state.router_model = "bypass"
                                req.state.router_reason = "tool_bypass_longpaste"
                                req.state.router_capability_level = "0"
                                req.state.router_pinned = "false"
                            except Exception:
                                pass
                        return {"tool_calls": []}
        except Exception:
            pass

    _debug_log_request(body)

    session_id = extract_session_id(req, body)
    user_id = extract_user_id(req, body)
    chat_id = extract_chat_id(req, body)
    session = await get_session_state(session_id)

    try:
        _msgs_lt_tool = list(body.get("messages") or [])
        _last_user_lt_tool = _last_user_text(_msgs_lt_tool)
        _lt_tokens_cur = _estimate_tokens(_last_user_lt_tool)
        _lt_ok_cur, _lt_reason_cur = _should_admit_longtext(_last_user_lt_tool, _lt_tokens_cur)
        _lt_is_new_doc = bool(_lt_ok_cur and _looks_like_summarize_request(_msgs_lt_tool))
        _lt_ack_only = bool(int(LONGTEXT_TOOL_ACK_BYPASS or 0) == 1 and _is_ack_only_turn(_last_user_lt_tool))
        _lt_has_active_summary = bool(str(session.get("longtext_tool_summary") or "").strip())
        _lt_has_image = _current_turn_has_image_payload(_msgs_lt_tool)
        _lt_is_internal = _detect_openwebui_internal_task(_last_user_lt_tool)
        _lt_is_code = _detect_code_or_deep(_last_user_lt_tool)

        if _lt_is_new_doc and not _lt_has_image and not _lt_is_internal and not _lt_is_code:
            _lt_doc_hash = _hash_text_for_session(_last_user_lt_tool)
            body["_scci_longtext_tool"] = 1
            body["_scci_longtext_tool_capture"] = 1
            body["_scci_longtext_tool_doc_hash"] = _lt_doc_hash
            body["_scci_lane_label"] = "longtext_tool"
            _lt_tool_msgs = _build_longtext_tool_messages(
                _msgs_lt_tool,
                char_budget=int(LONGTEXT_TOOL_DOC_MAX_CHARS or 120000),
            )
            if _lt_tool_msgs:
                body["messages"] = _lt_tool_msgs
                log.info(
                    f"[{req_id}] → SCCI LONGTEXT_TOOL sid={req_id} phase=ingest current_only=1 msgs={len(_lt_tool_msgs)} "
                    f"chars={len(_last_user_lt_tool or '')} reason={_lt_reason_cur}"
                )
        elif _lt_has_active_summary and not _lt_is_new_doc and not _lt_ack_only and not _lt_has_image and not _lt_is_internal:
            _lt_summary = str(session.get("longtext_tool_summary") or "").strip()
            if _lt_summary:
                _lt_base_system = _extract_leading_system_text(_msgs_lt_tool)
                body["messages"] = _build_longtext_followup_messages(_lt_summary, _last_user_lt_tool, base_system=_lt_base_system)
                body["_scci_longtext_tool"] = 1
                body["_scci_lane_label"] = "longtext_tool"
                log.info(
                    f"[{req_id}] → SCCI LONGTEXT_TOOL sid={req_id} phase=followup summary_only=1 "
                    f"summary_chars={len(_lt_summary)} user_chars={len(_last_user_lt_tool or '')}"
                )
        elif _lt_ack_only and _lt_has_active_summary:
            _lt_base_system = _extract_leading_system_text(_msgs_lt_tool)
            body["messages"] = _build_ack_only_messages(_last_user_lt_tool, base_system=_lt_base_system)
            body["_scci_lane_label"] = "direct"
            log.info(f"[{req_id}] → SCCI LONGTEXT_TOOL sid={req_id} phase=ack_bypass summary_injection=0 soft_reset=1")
    except Exception as _lt_tool_err:
        log.warning(f"[{req_id}] SCCI longtext_tool_prep_err={type(_lt_tool_err).__name__}:{_lt_tool_err}")

    _session_controller_prepare(session, session_id, "chat")
    pinned = _session_pinned_tuple(session)

    # allow forcing exact model: "[node] model"
    force_node = (req.headers.get("x-force-node") or "").strip()
    force_model = (req.headers.get("x-force-model") or "").strip()

    model_in = str(body.get("model") or "").strip()
    # v34: internal forced vision model (ctx ladder) so later selection cannot downgrade
    vision_forced_model = ""

    # Accept the same IDs we advertise in /v1/models: "SCCI-Router" and "[node] model" (space optional)
    direct_node, direct_model = parse_direct_model(model_in)
    if direct_node:
        nk = _normalize_node_key(direct_node)
        force_node = nk or direct_node.strip()
        force_model = direct_model.strip()

        # Strip the prefix from the payload early so upstream never sees "[node] ..."
        if force_model:
            body["model"] = force_model
            model_in = force_model

    if not model_in:
        body["model"] = "SCCI-Router"

    # -------- Direct node routing (OpenWebUI "[node] model" or x-force-node/model) --------
    # If a node is explicitly specified, bypass SCCI-Router capability routing and forward directly.
    if force_node:
        node_key = _normalize_node_key(force_node) or force_node
        node = _nodes.get(node_key) or _nodes.get(node_key.lower()) or _nodes.get(node_key.upper())
        if not node:
            return JSONResponse({"error": {"message": f"Unknown node '{force_node}'", "type": "invalid_request_error"}}, status_code=400)

        # Ensure we have fresh health/ready/model info (startup races otherwise cause forced 503)
        try:
            await check_node_health(node)
        except Exception:
            pass
        try:
            await check_node_ready(node)
        except Exception:
            pass
        try:
            await discover_models_for_node(node)
        except Exception:
            pass

        if node.in_cooldown() or (not node.healthy_cached()):
            return JSONResponse({"error": {"message": f"Node '{node.name}' is unavailable", "type": "server_error"}}, status_code=503)
        if not node.ready:
            return JSONResponse({"error": {"message": f"Node '{node.name}' is not ready", "type": "server_error"}}, status_code=503)

        # Decide which model to run on that node
        chosen = (force_model or "").strip()
        if not chosen:
            # If OpenWebUI sent only "[node]" with no model, pick best available for this request type.
            # (chat endpoint is text-only unless images are present)
            chosen = next(iter(node.models.keys()), "")
        if not chosen:
            return JSONResponse({"error": {"message": f"Node '{node.name}' has no models", "type": "server_error"}}, status_code=503)

        # Validate or resolve model id on that node
        if chosen not in node.models:
            resolved = find_model_by_hint(node, chosen) or find_model_by_hint(node, model_in)
            if resolved:
                chosen = resolved
            else:
                return JSONResponse({"error": {"message": f"Model '{chosen}' not found on node '{node.name}'", "type": "invalid_request_error"}}, status_code=404)

        # Upstream must receive the raw model id (without "[node]" prefix)
        body["model"] = chosen
    # v30: Token-estimation pre-check for huge summarize requests (avoid ctx overflow up front)
    if ROUTER_LONGPASTE_ENABLE:
        try:
            _msgs = body.get("messages", [])
            # v32.6 SICC-safe: detect image payloads BEFORE routing/sanitization (OpenAI-style + OpenWebUI variants)
            image_parts = []
            _msgs = body.get("messages", []) or []
            for _i, _m in enumerate(_msgs):
                c = _m.get("content") if isinstance(_m, dict) else None
                if isinstance(c, list):
                    _n = 0
                    for p in c:
                        if isinstance(p, dict) and p.get("type") in ("image_url", "input_image"):
                            _n += 1
                    if _n:
                        image_parts.append((_i, _n))
            # v34.2: distinguish "any images in history" vs "images in current user turn"
            has_image_history = bool(image_parts)
            _last_user_imgs = 0
            try:
                _lu = _msgs[-1] if _msgs else None
                _lc = (_lu or {}).get("content")
                if isinstance(_lc, list):
                    for _p in _lc:
                        if isinstance(_p, dict) and _p.get("type") in ("image_url", "input_image"):
                            _last_user_imgs += 1
            except Exception:
                pass
            has_image = (_last_user_imgs > 0)
            # Vision hard-override: image requests must not be forced onto pinned/non-vision models.
            if has_image:
                # Break pin if current pin is non-vision (SRVCC-style override)
                try:
                    if pinned and pinned[1] and ("vision" not in str(pinned[1]).lower()):
                        session["pinned_node"] = None
                        session["pinned_model"] = None
                        session["pinned_capability"] = None
                        pinned = (None, None)
                        log.info(f"[{req_id}] SICC unpin_nonvision_due_to_vision=1")
                except Exception:
                    pass

                if ROUTER_VISION_LADDER_ENABLE:
                    try:
                        _msgs = _trim_messages_for_vision(_msgs, ROUTER_VISION_HISTORY_TURNS)
                        _img_n = _count_images_in_messages(_msgs)
                        _all_txt = _extract_text_for_estimate(_msgs)
                        _txt_tokens = _estimate_tokens(_all_txt)
                        # v34.1: vision safety inflation (chars/4 underestimates real tokenizer)
                        try:
                            _txt_tokens = int(_txt_tokens * ROUTER_VISION_TEXT_MULT)
                        except Exception:
                            pass
                        _img_tokens = int(_img_n * ROUTER_VISION_TOKENS_PER_IMAGE)
                        _max_out = int(body.get('max_tokens') or body.get('max_completion_tokens') or 0)
                        _tokens_needed = int(_txt_tokens + _img_tokens + 64 + _max_out + ROUTER_CTX_COMPLETION_RESERVE)  # v34: include completion reserve
                        # overhead cushion

                        _vision_pick = _pick_vision_model_ladder(_tokens_needed)

                        # Force router to use the chosen 4B vision ctx variant
                        body["model"] = _vision_pick
                        model_in = _vision_pick
                        vision_forced_model = _vision_pick
                        # prevent direct-selection from stealing multimodal requests
                        direct_node = None
                        direct_model = None

                        if ROUTER_VISION_LADDER_LOG:
                            log.info(f"[{req_id}] vision_ladder pick={_vision_pick} tokens_needed~{_tokens_needed} img_n={_img_n} img_tokens={_img_tokens} txt_tokens={_txt_tokens}")
                    except Exception as _e:
                        try:
                            log.warning(f"[{req_id}] vision_ladder_failed err={type(_e).__name__}:{_e}")
                        except Exception:
                            pass

            # v32.6: image detection debug (default on; disable with ROUTER_LOG_IMAGE_DETECT=0)
            if ROUTER_LOG_IMAGE_DETECT:
                try:
                    if has_image:
                        parts_str = ",".join([f"{i}:{n}" for i, n in image_parts])
                        log.info(f"[{req_id}] image_detected=1 last_user_imgs={_last_user_imgs} history_imgs={1 if has_image_history else 0} parts={parts_str}")
                    else:
                        log.info(f"[{req_id}] image_detected=0 last_user_imgs={_last_user_imgs} history_imgs={1 if has_image_history else 0}")
                except Exception:
                    pass
            # v32.7: Force vision bucket as soon as images are detected (SICC-safe)
            if has_image and (ROUTER_VISION_OVERRIDE_ON_IMAGE or (ROUTER_VISION_TEXT_INTENT and _is_vision_description_prompt(last_user_txt))):
                forced_bucket = os.getenv("ROUTER_VISION_BUCKET_ON_IMAGE", "vision_fast")
                if forced_bucket and forced_bucket != "chat":
                    try:
                        log.info(f"[{req_id}] SICC vision_override=1 bucket={forced_bucket}")
                    except Exception:
                        pass
                    bucket = forced_bucket

            # v34.2: Policy A — text-only requests must not route to vision buckets
            if (not has_image) and str(bucket).startswith("vision"):
                bucket = "deep"
                try:
                    log.info(f"[{req_id}] SICC text_only -> rewrite bucket=deep (no vision override)")
                except Exception:
                    pass

            _utxt = _last_user_text(_msgs)
            if _looks_like_summarize_request(_msgs) and len(_utxt) >= ROUTER_LONGPASTE_CHAR_THRESHOLD:
                est_tok = _estimate_tokens_from_messages(_msgs)
                sel_ctx = _ctx_from_model_id(model_id, default_ctx=4096)
                if est_tok > int(sel_ctx * 0.9):
                    # Force non-tool chat routing on CPU0 (keep GPU/PHONE2 free)
                    bucket = "chat"
                    if "B-CPU0" in NODES:
                        try:
                            cpu_models = NODE_MODELS_CACHE.get("B-CPU0") or []
                            cand = []
                            for mid in cpu_models:
                                if FUNCTION_MODEL_PAT.search(str(mid)):
                                    continue
                                if VISION_MODEL_PAT.search(str(mid)):
                                    continue
                                cand.append(str(mid))
                            if cand:
                                cand.sort(key=lambda x: _ctx_from_model_id(x, 4096), reverse=True)
                                model_id = cand[0]
                                node_name = "B-CPU0"
                                reason = "longpaste_force_cpu_chat"
                                cap_level = 3
                        except Exception:
                            pass
        except Exception:
            pass
        # v32: post-route long-paste override (keeps B-PHONE2 out of big summarization flow)
        if ROUTER_LONGPASTE_ENABLE:
            try:
                _msgs = body.get("messages", []) if isinstance(body, dict) else []
                _utxt = _last_user_text(_msgs)
                if _looks_like_summarize_request(_msgs) and len(_utxt) >= ROUTER_LONGPASTE_CHAR_THRESHOLD:
                    # Force to CPU chat (non-tool, non-vision) to avoid tool bucket and PHONE2
                    bucket = "chat"
                    if "B-CPU0" in NODES:
                        node_name = "B-CPU0"
                        try:
                            cpu_models = NODE_MODELS_CACHE.get("B-CPU0") or []
                            cand = []
                            for mid in cpu_models:
                                if FUNCTION_MODEL_PAT.search(str(mid)):
                                    continue
                                if VISION_MODEL_PAT.search(str(mid)):
                                    continue
                                cand.append(str(mid))
                            if cand:
                                cand.sort(key=lambda x: _ctx_from_model_id(x, 4096), reverse=True)
                                model_id = cand[0]
                        except Exception:
                            pass
                    cap_level = max(int(cap_level or 0), 3)
                    reason = "longpaste_force_cpu_chat"
            except Exception:
                pass

        

    # v10.0: policy-table routing (first match wins)
    try:
        _policy_ctx = {
            "bucket": bucket,
            "has_image": bool(has_image),
            "medical_intent": bool(_is_medical_intent_text(_last_user_text(body.get("messages") or []))),
        }
        _policy_name, _policy_prio, _policy_action = _evaluate_scci_priority_policy(_policy_ctx)
        if _policy_name:
            log.info(f"[{req_id}] → SCCI POLICY_MATCH sid={req_id} policy={_policy_name} priority={_policy_prio}")
            if isinstance(_policy_action, dict):
                if _policy_action.get("bucket"):
                    bucket = str(_policy_action.get("bucket"))
                if _policy_action.get("node"):
                    node_name = str(_policy_action.get("node"))
                if _policy_action.get("model"):
                    model_id = str(_policy_action.get("model"))
                if _policy_name == "medical_vision" and node_name and model_id:
                    reason = "policy_medical_vision"
                    log.info(f"[{req_id}] → SCCI MEDICAL_ROUTE sid={req_id} node={node_name} model={model_id}")
    except Exception as _policy_err:
        log.warning(f"[{req_id}] scci_policy_err={type(_policy_err).__name__}:{_policy_err}")

    # v32: safety guard - never route vision/tool buckets to a non-capable node
    if not node_name or not model_id:
        # Provide detailed eligibility logging for faster debugging
        try:
            want_bucket = bucket
            if want_bucket.startswith("vision"):
                target_nodes = [x.strip() for x in os.getenv("ROUTER_VISION_NODES", "B-GPU0,B-CPU0").split(",") if x.strip()]
                want_model = VISION_FAST_MODEL_ID if want_bucket == "vision_fast" else VISION_REASON_MODEL_ID
                elig = []
                for nn in target_nodes:
                    n0 = _nodes.get(nn)
                    if not n0:
                        elig.append(f"{nn}:missing")
                        continue
                    mids = [str(m.get("id")) for m in (n0.models_cache or []) if isinstance(m, dict)]
                    has = any(m == want_model for m in mids)
                    elig.append(f"{nn}:{'ok' if has else 'no_model'}")
                log.warning(f"[{req_id}] no_eligible_candidates bucket={want_bucket} want_model={want_model} elig={elig}")
            elif want_bucket == "tool":
                target_nodes = [x.strip() for x in os.getenv("ROUTER_TOOL_NODES", "B-PHONE2,B-CPU0").split(",") if x.strip()]
                want_model = TOOL_MODEL_ID
                elig = []
                for nn in target_nodes:
                    n0 = _nodes.get(nn)
                    if not n0:
                        elig.append(f"{nn}:missing")
                        continue
                    mids = [str(m.get('id')) for m in (n0.models_cache or []) if isinstance(m, dict)]
                    has = any(m == want_model for m in mids)
                    elig.append(f"{nn}:{'ok' if has else 'no_model'}")
                log.warning(f"[{req_id}] no_eligible_candidates bucket={want_bucket} want_model={want_model} elig={elig}")
        except Exception:
            pass

        if bucket.startswith("vision"):
            # Vision requests must go to GPU/CPU vision-capable nodes
            fb_nodes = [x.strip() for x in os.getenv("ROUTER_VISION_NODES", "B-GPU0,B-CPU0").split(",") if x.strip()]
            fb_model = VISION_FAST_MODEL_ID if bucket == "vision_fast" else VISION_REASON_MODEL_ID
            for fb in fb_nodes:
                n = _nodes.get(fb)
                if not n:
                    continue
                # ensure model exists on that node
                mids = [str(m.get("id")) for m in (getattr(n, 'models_cache', None) or getattr(n, 'models', []) or []) if isinstance(m, dict)]
                if fb_model not in mids:
                    continue
                node_name, model_id = n.name, fb_model
                log.warning(f"[{req_id}] routing_fallback bucket={bucket} -> {node_name}:{model_id}")

        elif bucket == "tool":
            # Tool bucket: XPZ first, CPU fallback
            fb_nodes = [x.strip() for x in os.getenv("ROUTER_TOOL_NODES", "B-PHONE2,B-CPU0").split(",") if x.strip()]
            for fb in fb_nodes:
                n = _nodes.get(fb)
                if not n:
                    continue
                mids = [str(m.get("id")) for m in (getattr(n, 'models_cache', None) or getattr(n, 'models', []) or []) if isinstance(m, dict)]
                if TOOL_MODEL_ID not in mids:
                    continue
                node_name, model_id = n.name, TOOL_MODEL_ID
                log.warning(f"[{req_id}] routing_fallback bucket=tool -> {node_name}:{model_id}")
                break

        else:
            # Chat/general seed selection (phone0 first, CPU0 fallback)
            fb_nodes = [x.strip() for x in os.getenv("ROUTER_CHAT_FALLBACK_NODES", "B-PHONE0,B-CPU0").split(",") if x.strip()]
            for fb in fb_nodes:
                n = _nodes.get(fb)
                if not n:
                    continue
                preferred_model_id = bucket_hint(fb, bucket) or CHAT_MODEL_ID
                resolved_model_id = resolve_model_name_for_node(req_id, n, preferred_model_id, preferred_model_id)
                node_name, model_id = n.name, resolved_model_id
                if preferred_model_id != resolved_model_id:
                    log.info(f"[{req_id}] routing_seed bucket={bucket} node={node_name} preferred_model={preferred_model_id} resolved_model={resolved_model_id}")
                else:
                    log.info(f"[{req_id}] routing_seed bucket={bucket} node={node_name} model={resolved_model_id}")
                break


    # v2.2: early long-text chat admission using the chat ladder.
    try:
        _msgs_long = body.get("messages") or []
        _orig_msgs_long = list(_msgs_long)
        _has_image_payload_long = _messages_have_image_payload(_msgs_long)
        _last_user_long = _last_user_text(_msgs_long)
        _is_openwebui_internal_long = _detect_openwebui_internal_task(_last_user_long)
        _is_code_or_deep_long = _detect_code_or_deep(_last_user_long)
        _longtext_chars = len(_last_user_long or "")
        _tokens_needed_long = _estimate_tokens(_last_user_long) + int(ROUTER_CTX_COMPLETION_RESERVE)
        _longtext_ok, _longtext_reason = _should_admit_longtext(_last_user_long, _tokens_needed_long)

        _longtext_authoritative = (
            (model_in == "SCCI-Router" or not model_in)
            and str(bucket) in ("chat", "direct")
            and not bool(body.get("_scci_longtext_tool"))
            and not _has_image_payload_long
            and not _is_openwebui_internal_long
            and not _is_code_or_deep_long
            and _longtext_ok
        )

        if (
            (model_in == "SCCI-Router" or not model_in)
            and not _has_image_payload_long
            and not _is_openwebui_internal_long
            and not _is_code_or_deep_long
            and _longtext_chars >= int(LONGTEXT_DIRECT_THRESHOLD or 1500)
            and not _longtext_ok
            and _longtext_reason not in ("below_threshold", "empty")
        ):
            log.info(
                f"[{req_id}] → SCCI LONGTEXT_SKIP sid={req_id} chars={_longtext_chars} "
                f"tokens~{int(_tokens_needed_long or 0)} reason={_longtext_reason}"
            )

        if _longtext_authoritative:
            try:
                (
                    _tokens_needed_long,
                    _chat_target_model,
                    _chat_target_node,
                    _chat_target_ctx,
                    _chat_ratio,
                ) = _select_longtext_ladder_from_final_body(
                    req_id=req_id,
                    body=body,
                    fallback_msgs=_msgs_long,
                    fallback_tokens=int(_tokens_needed_long or 0),
                )
            except Exception as _final_payload_ctx_err:
                log.warning(f"[{req_id}] SCCI final_payload_ctx_err={type(_final_payload_ctx_err).__name__}:{_final_payload_ctx_err}")
        _last_user_long_len = len(_last_user_long or "")
        _is_openwebui_internal_long = _detect_openwebui_internal_task(_last_user_long)
        _is_code_or_deep_long = _detect_code_or_deep(_last_user_long)

        if (
            (model_in == "SCCI-Router" or not model_in)
            and not bool(body.get("_scci_longtext_tool"))
            and not direct_node
            and not str(bucket).startswith("vision")
            and not _has_image_payload_long
            and not _is_openwebui_internal_long
            and not _is_code_or_deep_long
            and _longtext_ok
        ):
            (
                _tokens_needed_long,
                _chat_target_model,
                _chat_target_node,
                _chat_target_ctx,
                _chat_ratio,
            ) = _select_longtext_ladder_from_final_body(
                req_id=req_id,
                body=body,
                fallback_msgs=_msgs_long,
                fallback_tokens=int(_tokens_needed_long or 0),
            )
            log.info(
                f"[{req_id}] → SCCI LONGTEXT_INTENT sid={req_id} longtext=1 chars={_last_user_long_len} "
                f"tokens~{int(_tokens_needed_long or 0)} target={_chat_target_node} model={_chat_target_model} reason={_longtext_reason}"
            )
            log.info(
                f"[{req_id}] → SCCI CHAT_LADDER sid={req_id} pick={_chat_target_model} "
                f"node={_chat_target_node} ctx={int(_tokens_needed_long or 0)}/{_chat_target_ctx} ratio={_chat_ratio:.2f}"
            )
            node_name = _chat_target_node
            model_id = _chat_target_model
            bucket = "chat"
            body["model"] = _chat_target_model
            body["_scci_forced_chat_ladder"] = 1
            try:
                _rebuilt_long_msgs = _build_longtext_messages(
                    _msgs_long,
                    keep_last_n=LONGTEXT_HISTORY_MESSAGES,
                    char_budget=LONGTEXT_HISTORY_CHAR_BUDGET,
                )
                if _rebuilt_long_msgs:
                    body["messages"] = _rebuilt_long_msgs
                    log.info(
                        f"[{req_id}] → SCCI LONGTEXT_CTX sid={req_id} kept_msgs={len(_rebuilt_long_msgs)} mode=system_history one_shot=1 "
                        f"history_n={LONGTEXT_HISTORY_MESSAGES} history_char_budget={LONGTEXT_HISTORY_CHAR_BUDGET}"
                    )
            except Exception as _ctx_rebuild_err:
                log.warning(f"[{req_id}] SCCI longtext_ctx_rebuild_err={type(_ctx_rebuild_err).__name__}:{_ctx_rebuild_err}")
            try:
                session["pinned_node"] = node_name
                session["pinned_model"] = model_id
                session["pinned_capability"] = 0
            except Exception:
                pass
    except Exception as _longtext_early_err:
        log.warning(f"[{req_id}] SCCI longtext_early_detect_err={type(_longtext_early_err).__name__}:{_longtext_early_err}")

    # v2.7: on multimodal requests, keep the original payload untouched so
    # vision templates in llama.cpp/OpenWebUI are not polluted by longtext history rewriting.
    try:
        _msgs_vs = body.get("messages") or []
        if _current_turn_has_image_payload(_msgs_vs):
            if "_scci_forced_chat_ladder" in body:
                body.pop("_scci_forced_chat_ladder", None)
            if "_rebuilt_long_msgs" in locals():
                body["messages"] = _orig_msgs_long if "_orig_msgs_long" in locals() else _msgs_vs
            try:
                _vision_current_turn_only = _current_turn_only_messages(_msgs_vs)
                if _vision_current_turn_only:
                    _prior_dropped = max(0, len(_msgs_vs) - len(_vision_current_turn_only))
                    body["messages"] = _vision_current_turn_only
                    log.info(f"[{req_id}] → SCCI VISION_ISOLATE sid={req_id} current_turn_only=1 prior_msgs_dropped={_prior_dropped}")
            except Exception as _vision_isolate_err:
                log.warning(f"[{req_id}] SCCI vision_isolate_err={type(_vision_isolate_err).__name__}:{_vision_isolate_err}")
            log.info(f"[{req_id}] → SCCI VISION_SAFE sid={req_id} original_multimodal_payload=1")
    except Exception as _vision_safe_err:
        log.warning(f"[{req_id}] SCCI vision_safe_err={type(_vision_safe_err).__name__}:{_vision_safe_err}")

    try:
        _msgs_vprep = body.get("messages") or []
        if int(ROUTER_VISION_PREPROCESS_ENABLE or 0) == 1 and _current_turn_has_image_payload(_msgs_vprep):
            if Image is None:
                log.warning(f"[{req_id}] → SCCI VISION_PREP_BYPASS sid={req_id} cause=pillow_unavailable")
            else:
                _prepped_msgs, _pre_seen, _pre_rewritten = _prepare_multimodal_images_square(
                    _msgs_vprep,
                    size=int(ROUTER_VISION_PREPROCESS_SIZE or 896),
                )
                if _pre_seen > 0 and _pre_rewritten > 0:
                    body["messages"] = _prepped_msgs
                    log.info(
                        f"[{req_id}] → SCCI VISION_PREP sid={req_id} resize={int(ROUTER_VISION_PREPROCESS_SIZE or 896)}x{int(ROUTER_VISION_PREPROCESS_SIZE or 896)} "
                        f"mode=letterbox bg={ROUTER_VISION_PREPROCESS_BG} rewritten={_pre_rewritten}/{_pre_seen}"
                    )
                elif _current_turn_has_image_payload(_msgs_vprep):
                    log.info(
                        f"[{req_id}] → SCCI VISION_PREP_BYPASS sid={req_id} cause=no_supported_data_url_images"
                    )
    except Exception as _vision_prep_err:
        log.warning(f"[{req_id}] SCCI vision_prep_err={type(_vision_prep_err).__name__}:{_vision_prep_err}")

    # Hardened precedence matrix: vision > file_code > coding > tools > ui_helper > chat
    try:
        _msgs_lane = body.get("messages") or []
        _last_user_lane = _last_user_text(_msgs_lane)
        _attached_code_files = _detect_attached_code_files(body if isinstance(body, dict) else {}, _msgs_lane)
        _file_code_intent = bool(_attached_code_files) and _is_file_code_intent(_last_user_lane)
        _coding_text_signal = _detect_code_or_deep(_last_user_lane)
        _tool_signal = _messages_have_tool_prompt(_msgs_lane)
        _helper_meta_lane = _classify_openwebui_internal_task(_last_user_lane)
        _ui_helper_signal = bool(_helper_meta_lane.get("is_helper", False))
        _current_turn_has_image = _current_turn_has_image_payload(_msgs_lane)
        _helper_conf = max(0.0, min(1.0, float((_helper_meta_lane.get("score", 0) or 0)) / 10.0))
        _intent_flags = {
            "vision": bool(_current_turn_has_image),
            "file_code": bool(_file_code_intent),
            "coding": bool(_coding_text_signal or _file_code_intent),
            "tool": bool(_tool_signal),
            "ui_helper": bool(_ui_helper_signal),
        }
        _intent_winner, _intent_overridden = _winner_from_intent_matrix(_intent_flags)
        _intent_detail = f"files={','.join(_attached_code_files) if _attached_code_files else '-'} helper_score={int(_helper_meta_lane.get('score', 0) or 0)} confidence={_helper_conf:.2f}"
        _scci_log_intent_matrix(req_id, _intent_flags, _intent_winner, _intent_overridden, _intent_detail)
        if _intent_overridden:
            log.info(f"[{req_id}] → SCCI INTENT_OVERRIDE sid={req_id} winner={_intent_winner} suppressed={','.join(_intent_overridden)}")
        body["_scci_intent_winner"] = _intent_winner
        body["_scci_file_code_force"] = 1 if _file_code_intent else 0
        body["_scci_code_file_names"] = list(_attached_code_files)
        body["_scci_ui_helper_allowed"] = 1 if (_intent_winner == "ui_helper") else 0
        if _file_code_intent:
            log.info(f"[{req_id}] → SCCI LANE_OVERRIDE sid={req_id} forced_lane=coding cause=file_extension files={','.join(_attached_code_files)} context=last_user")
    except Exception as _intent_matrix_err:
        log.warning(f"[{req_id}] SCCI intent_matrix_err={type(_intent_matrix_err).__name__}:{_intent_matrix_err}")

    # v2.0+: early OpenWebUI internal helper-task admission to PHONE1 UI helper lane with CPU fallback.
    # This keeps FunctionGemma reserved for real demo tools such as time/date.
    try:
        _msgs_tool = body.get("messages") or []
        _last_user_tool = _last_user_text(_msgs_tool)
        _helper_meta = _classify_openwebui_internal_task(_last_user_tool)
        _is_openwebui_internal = bool(_helper_meta.get("is_helper", False))
        if _is_openwebui_internal and int(body.get("_scci_ui_helper_allowed", 0) or 0) != 1:
            _log_ui_helper(req_id, f"bypass=1 cause=precedence winner={body.get('_scci_intent_winner') or 'chat'}")
        if (model_in == "SCCI-Router" or not model_in) and not direct_node and not str(bucket).startswith("vision") and _is_openwebui_internal and int(body.get("_scci_ui_helper_allowed", 0) or 0) == 1:
            _tool_trim_chars = 0
            _tool_eval_msgs = list(_msgs_tool)
            try:
                if int(TOOLS_TRIM_ENABLED or 0) == 1:
                    _trimmed_tool_msgs = _trim_openwebui_helper_messages(_msgs_tool, max_chars=UI_HELPER_TRIM_MAX_CHARS, chat_history_max_chars=UI_HELPER_CHAT_HISTORY_MAX_CHARS)
                    if _trimmed_tool_msgs:
                        _tool_trim_chars = max(0, len(_last_user_tool or "") - len(_extract_text_from_content(_trimmed_tool_msgs[-1].get("content")) or ""))
                        body["messages"] = _trimmed_tool_msgs
                        _tool_eval_msgs = _trimmed_tool_msgs
            except Exception as _tool_trim_err:
                log.warning(f"[{req_id}] SCCI ui_helper_trim_err={type(_tool_trim_err).__name__}:{_tool_trim_err}")

            _tool_tokens = _estimate_tokens_from_messages(_tool_eval_msgs) + int(ROUTER_CTX_COMPLETION_RESERVE or 0)
            _tool_ctx = int(_parse_ctx_from_model_id(UI_HELPER_MODEL_ID) or 2048)
            _tool_safe_limit = _tool_ctx_safe_limit(UI_HELPER_MODEL_ID)
            _tool_overflow_guard = int(_tool_tokens or 0) > int(_tool_safe_limit or 0)
            _log_ui_helper(
                req_id,
                f"intent=internal_ui_task score={int(_helper_meta.get('score', 0) or 0)} "
                f"signals={','.join(_helper_meta.get('signals', []) or []) or '-'} "
                f"target={UI_HELPER_NODE_PRIMARY} model={UI_HELPER_MODEL_ID} "
                f"trimmed={1 if _tool_trim_chars > 0 else 0} trim_chars={_tool_trim_chars}"
            )

            if int(ROUTER_TOOL_LARGE_PROMPT_CHAT_FALLBACK or 0) == 1 and _tool_overflow_guard:
                (
                    _tool_chat_tokens,
                    _tool_chat_model,
                    _tool_chat_node,
                    _tool_chat_ctx,
                    _tool_chat_ratio,
                ) = _select_longtext_ladder_from_final_body(
                    req_id=req_id,
                    body=body,
                    fallback_msgs=_tool_eval_msgs,
                    fallback_tokens=int(_tool_tokens or 0),
                )
                log.info(
                    f"[{req_id}] → SCCI UI_HELPER_BYPASS sid={req_id} reason=ctx_guard "
                    f"helper_tokens~={int(_tool_tokens or 0)} safe_limit={_tool_safe_limit} "
                    f"fallback_node={_tool_chat_node} fallback_model={_tool_chat_model} "
                    f"fallback_ctx={int(_tool_chat_tokens or 0)}/{_tool_chat_ctx} ratio={_tool_chat_ratio:.2f}"
                )
                node_name = _tool_chat_node
                model_id = _tool_chat_model
                bucket = "chat"
                body["model"] = _tool_chat_model
                body.pop("_scci_forced_ui_helper_lane", None)
                body["_scci_forced_chat_ladder"] = 1
                try:
                    session["pinned_node"] = node_name
                    session["pinned_model"] = model_id
                    session["pinned_capability"] = 0
                except Exception:
                    pass
            else:
                _tool_chosen_node = None
                _tool_reason = "ui_helper_primary"
                for _nn in [UI_HELPER_NODE_PRIMARY, UI_HELPER_NODE_FALLBACK]:
                    _n = _nodes.get(_nn)
                    if not _n:
                        continue
                    try:
                        await check_node_health(_n)
                        await discover_models_for_node(_n)
                    except Exception:
                        pass
                    if getattr(_n, "models", None) is None or UI_HELPER_MODEL_ID in getattr(_n, "models", {}):
                        _tool_chosen_node = _nn
                        _tool_reason = "ui_helper_primary" if _nn == UI_HELPER_NODE_PRIMARY else "ui_helper_fallback_cpu"
                        break
                if _tool_chosen_node:
                    _log_ui_helper(
                        req_id,
                        f"lane=ui_helper primary={UI_HELPER_NODE_PRIMARY} selected={_tool_chosen_node} "
                        f"reason={_tool_reason} model={UI_HELPER_MODEL_ID}"
                    )
                    node_name = _tool_chosen_node
                    model_id = UI_HELPER_MODEL_ID
                    bucket = "ui_helper"
                    body["model"] = UI_HELPER_MODEL_ID
                    body["_scci_forced_ui_helper_lane"] = 1
                    try:
                        session["pinned_node"] = node_name
                        session["pinned_model"] = model_id
                        session["pinned_capability"] = 0
                    except Exception:
                        pass
    except Exception as _tool_early_err:
        log.warning(f"[{req_id}] SCCI ui_helper_early_detect_err={type(_tool_early_err).__name__}:{_tool_early_err}")


    # v9 coding: early task-scoped coding admission before fast-path/direct fallback.
    try:
        _msgs_coding = body.get("messages") or []
        _last_user_coding = _last_user_text(_msgs_coding)
        _code_file_names = list(body.get("_scci_code_file_names") or [])
        _file_code_forced = int(body.get("_scci_file_code_force", 0) or 0) == 1
        _is_code_or_deep = _detect_code_or_deep(_last_user_coding) or _file_code_forced
        _coding_tool_info = {"trimmed": 0, "orig_messages": len(_msgs_coding), "new_messages": len(_msgs_coding), "orig_chars": len(_extract_text_for_estimate(_msgs_coding) or ""), "new_chars": len(_extract_text_for_estimate(_msgs_coding) or "")}
        if _is_code_or_deep:
            _msgs_coding, _coding_tool_info = _build_coding_tool_messages(_msgs_coding, char_budget=int(CODING_TOOL_CHAR_BUDGET or 12000))
            body["messages"] = _msgs_coding
            _last_user_coding = _last_user_text(_msgs_coding)
        _all_text_coding = _extract_text_for_estimate(_msgs_coding)
        _tokens_needed_coding = _estimate_tokens(_all_text_coding) + int(ROUTER_CTX_COMPLETION_RESERVE)
        _force_chat_lane = 0
        try:
            _lu_guard = (_last_user_coding or "").lower()
            if ("summarize" in _lu_guard) or ("summary" in _lu_guard) or ("wikipedia" in _lu_guard):
                _force_chat_lane = 1
        except Exception:
            _force_chat_lane = 0
        _is_large_input = len(_last_user_coding or "") >= 2000

        if (model_in == "SCCI-Router" or not model_in) and not direct_node and not str(bucket).startswith("vision"):
            log.info(
                f"[{req_id}] → SCCI INTENT sid={req_id} coding={1 if _is_code_or_deep else 0} file_code={1 if _file_code_forced else 0} "
                f"large_input={1 if _is_large_input else 0}"
            )
            if _is_code_or_deep:
                log.info(
                    f"[{req_id}] → SCCI CODING_TOOL_SCOPE  # v9.7.7: ensure classification uses full user text before trimming sid={req_id} mode=last_user_only trimmed={int(_coding_tool_info.get('trimmed',0) or 0)} "
                    f"msgs={int(_coding_tool_info.get('orig_messages',0) or 0)}->{int(_coding_tool_info.get('new_messages',0) or 0)} "
                    f"chars={int(_coding_tool_info.get('orig_chars',0) or 0)}->{int(_coding_tool_info.get('new_chars',0) or 0)}"
                )

        if (model_in == "SCCI-Router" or not model_in) and not direct_node and not str(bucket).startswith("vision") and int(body.get("_scci_forced_tool_lane", 0) or 0) != 1 and int(_force_chat_lane or 0) != 1 and _is_code_or_deep and str(body.get("_scci_intent_winner") or "chat") in ("coding", "file_code"):
            user_text_cls = _last_user_text(_msgs_coding)
            try:
                fast_label = await _v976_run_classifier(req_id, user_text_cls)
                _classification = fast_label
                _class_meta = {"source": "v976_fast", "classifier_label": fast_label}
            except Exception:
                _classification, _class_meta = await _classify_coding_complexity(req_id, _msgs_coding, file_code_forced=_file_code_forced)
            _user_text_coding = _last_user_text(_msgs_coding)
            _reasoning_level = _v974_reasoning_level(_user_text_coding)

            # Surgical correction for guarded over-downgrade and classifier timeouts:
            # keep easy/trivial fixes simple, but promote architecture / implementation requests.
            if _classification == "simple_code" and _v973_should_promote_simple(_user_text_coding, _class_meta):
                _classification = "complex_code" if _reasoning_level == "high" else "medium_code"
                _class_meta["source"] = f"{_class_meta.get('source')}+v974_promote"

            # If the classifier timed out on CPU0, trust strong regex reasoning as backup.
            if str(_class_meta.get("source") or "").startswith("heuristic_exc_ReadTimeout"):
                if _reasoning_level == "high":
                    _classification = "complex_code"
                    _class_meta["source"] = f"{_class_meta.get('source')}+v974_timeout_promote_high"
                elif _reasoning_level == "medium" and _classification == "simple_code":
                    _classification = "medium_code"
                    _class_meta["source"] = f"{_class_meta.get('source')}+v974_timeout_promote_medium"

            _target_coder_model = _pick_coding_model_by_classification(_classification, int(_tokens_needed_coding or 0))
            if _classification == "medium_code" and _reasoning_level in ("medium", "high"):
                _target_coder_model = QWEN35_4B_MODEL
            elif _classification == "complex_code" and _reasoning_level != "high":
                _target_coder_model = QWEN35_4B_MODEL

            log.info(
                f"[{req_id}] → SCCI CODING_CLASSIFY sid={req_id} final={_classification} heuristic={_class_meta.get('heuristic_label')} llm={_class_meta.get('classifier_label')} source={_class_meta.get('source')}"
            )
            log.info(
                f"[{req_id}] → SCCI CODER_LADDER sid={req_id} pick={_target_coder_model} tokens_needed~{int(_tokens_needed_coding or 0)}"
            )

            _chosen_coder_node, _resolved_coder_model, _sched_reason = await _v973_select_coding_node_and_model(
                req_id, _classification, _reasoning_level, int(_tokens_needed_coding or 0)
            )

            if _chosen_coder_node and _resolved_coder_model:
                log.info(
                    f"[{req_id}] → SCCI SCHED sid={req_id} lane=coding class={_classification} primary={_coding_candidate_nodes(_classification)[0]} "
                    f"selected={_chosen_coder_node} reason={_sched_reason} model={_resolved_coder_model}"
                )
                log.info(
                    f"[{req_id}] → SCCI V974 sid={req_id} reasoning={_reasoning_level} "
                    f"selected={_resolved_coder_model} node={_chosen_coder_node}"
                )
                if _file_code_forced:
                    log.info(f"[{req_id}] → SCCI FILE_CODE_ROUTE sid={req_id} force=1 lane=coding files={','.join(_code_file_names) if _code_file_names else '-'} context=last_user")
                node_name = _chosen_coder_node
                model_id = _resolved_coder_model
                bucket = "coding"
                body["model"] = _resolved_coder_model
                body["_scci_forced_coding_lane"] = 1
                body["_scci_coding_classification"] = _classification
                body["_scci_reasoning_level"] = _reasoning_level
                trace_ctx_needed = int(_tokens_needed_coding or 0)
                trace_phone_ctx = int(_parse_ctx_from_model_id(_resolved_coder_model) or 4096)
                trace_ratio = float(trace_ctx_needed) / float(max(1, trace_phone_ctx))
                try:
                    session["pinned_node"] = node_name
                    session["pinned_model"] = model_id
                    session["pinned_capability"] = 2 if _classification != "complex_code" else 3
                except Exception:
                    pass
    except Exception as _coding_early_err:
        log.warning(f"[{req_id}] SCCI coding_early_detect_err={type(_coding_early_err).__name__}:{_coding_early_err}")


    # Optional fast-path: only for non-vision buckets (vision must go through vision candidate selection)
    if node_name and model_id and not str(bucket).startswith("vision"):
        # v34.11: chat-only predictive SRVCC (PHONE0 -> CPU0) with PREWARM + HANDOVER + TRACE logs
        fast_bucket = "direct"
        fast_reason = "direct_node_selection"
        trace_ctx_needed = None
        trace_phone_ctx = None
        trace_ratio = None
        trace_prewarm = 0
        trace_handover = 0
        try:
            # only for chat/micro buckets, and only when we are about to serve on B-PHONE0
            if bucket in ("chat", "micro") and _normalize_node_key(node_name) in ("B-PHONE0", "B_PHONE0"):
                phone_ctx = int(node_max_ctx("B-PHONE0") or 0) or 2048
                ctx_needed = int(prompt_tok_est) + int(max_tokens) + int(ROUTER_CTX_COMPLETION_RESERVE)
                ratio = float(ctx_needed) / float(max(1, phone_ctx))
                trace_ctx_needed = ctx_needed
                trace_phone_ctx = phone_ctx
                trace_ratio = ratio

                # PREWARM CPU0 when near ctx limit (non-blocking, rate-limited per session)
                if ROUTER_PREWARM_ENABLED and ratio >= ROUTER_CTX_PREWARM_RATIO and ratio < ROUTER_CTX_HANDOVER_RATIO:
                    now = time.time()
                    last_pw = float(session.get("prewarm_cpu0_ts") or 0.0)
                    if (now - last_pw) >= float(ROUTER_PREWARM_COOLDOWN_S):
                        session["prewarm_cpu0_ts"] = now
                        _target_model = _pick_chat_ladder_model_stepup(ctx_needed, phone_ctx)
                        _target_node_name = _pick_chat_ladder_node(_target_model)
                        _target_node = _nodes.get(_target_node_name)
                        if _target_node:
                            trace_prewarm = 1
                            session["prewarmed_target"] = _target_node_name
                            session["prewarmed_model"] = _target_model
                            session["bearer_state"] = "PREWARMED"
                            log.info(f"[{req_id}] → SCCI PREWARM sid={req_id} phone0→{_target_node_name.lower().replace('b-','')} ctx={ctx_needed}/{phone_ctx} ratio={ratio:.2f} model={_target_model}")
                            asyncio.create_task(_prewarm_node_model(req_id, _target_node_name, _target_node, _target_model))

                # Predictive HANDOVER using chat ladder when ctx crosses handover ratio (or if PHONE0 doesn't fit)
                if ratio >= ROUTER_CTX_HANDOVER_RATIO or (not ctx_fits("B-PHONE0", prompt_tok_est, max_tokens)):
                    _target_model = _pick_chat_ladder_model_stepup(ctx_needed, phone_ctx)
                    _target_node_name = _pick_chat_ladder_node(_target_model)
                    _target_node = _nodes.get(_target_node_name)
                    if _target_node:
                        prev = "B-PHONE0"
                        node_name = _target_node_name
                        model_id = _target_model
                        fast_bucket = "chat"
                        fast_reason = "ctx_predictive_handover"
                        trace_handover = 1
                        log.warning(f"[{req_id}] → SCCI HANDOVER sid={req_id} phone0->{_target_node_name.lower().replace('b-','')} cause=ctx_predictive ctx={ctx_needed}/{phone_ctx} ratio={ratio:.2f} model={_target_model}")
                        # Pin session to selected ladder target for stability after ctx escalation
                        try:
                            session["pinned_node"] = _target_node_name
                            session["pinned_model"] = model_id
                            session["pinned_capability"] = 2
                        except Exception:
                            pass
        except Exception:
            pass

        # v34.13: predictive ctx guard must run before final direct route is logged.
        # If earlier estimation block was skipped / undercounted, do a final KISS check here for chat/micro on PHONE0.
        try:
            if bucket in ("chat", "micro") and _normalize_node_key(node_name) in ("B-PHONE0", "B_PHONE0"):
                _msgs_for_ctx = body.get("messages") if isinstance(body, dict) else msgs
                _txt = _extract_text_for_estimate(_msgs_for_ctx)
                _est = _estimate_tokens(_txt)
                _ctx_current = int(_parse_ctx_from_model_id(model_id) or node_max_ctx("B-PHONE0") or 2048)
                _mx = int((body.get("max_tokens") if isinstance(body, dict) else None) or DEFAULT_MAX_TOKENS)
                _ctx_needed = int(_est) + int(_mx) + int(ROUTER_CTX_COMPLETION_RESERVE)
                _ratio = float(_ctx_needed) / float(max(1, _ctx_current))
                trace_ctx_needed = _ctx_needed
                trace_phone_ctx = _ctx_current
                trace_ratio = _ratio

                if ROUTER_PREWARM_ENABLED and _ratio >= ROUTER_CTX_PREWARM_RATIO and _ratio < ROUTER_CTX_HANDOVER_RATIO:
                    _now = time.time()
                    _last_pw = float(session.get("prewarm_cpu0_ts") or 0.0)
                    if (_now - _last_pw) >= float(ROUTER_PREWARM_COOLDOWN_S):
                        session["prewarm_cpu0_ts"] = _now
                        _target_model = _pick_chat_ladder_model_stepup(_ctx_needed, _ctx_current)
                        _target_node_name = _pick_chat_ladder_node(_target_model)
                        _target_node = _nodes.get(_target_node_name)
                        if _target_node:
                            trace_prewarm = 1
                            session["prewarmed_target"] = _target_node_name
                            session["prewarmed_model"] = _target_model
                            session["bearer_state"] = "PREWARMED"
                            log.info(f"[{req_id}] → SCCI PREWARM sid={req_id} phone0→{_target_node_name.lower().replace('b-','')} ctx={_ctx_needed}/{_ctx_current} ratio={_ratio:.2f} model={_target_model}")
                            asyncio.create_task(_prewarm_node_model(req_id, _target_node_name, _target_node, _target_model))

                # predictive handover at request start only; bearer lock handles stream continuity after route starts
                if _ratio >= ROUTER_CTX_HANDOVER_RATIO or _ctx_needed >= _ctx_current:
                    _target_model = _pick_chat_ladder_model_stepup(_ctx_needed, _ctx_current)
                    _target_node_name = _pick_chat_ladder_node(_target_model)
                    _target_node = _nodes.get(_target_node_name)
                    if _target_node:
                        trace_handover = 1
                        node_name = _target_node_name
                        model_id = _target_model
                        body['model'] = model_id
                        fast_bucket = "chat"
                        fast_reason = "ctx_predictive_handover"
                        log.warning(f"[{req_id}] → SCCI HANDOVER sid={req_id} phone0->{_target_node_name.lower().replace('b-','')} cause=ctx_predictive ctx={_ctx_needed}/{_ctx_current} ratio={_ratio:.2f} model={_target_model}")
                        try:
                            session["pinned_node"] = _target_node_name
                            session["pinned_model"] = model_id
                            session["pinned_capability"] = 2
                        except Exception:
                            pass
        except Exception as e:
            log.warning(f"[{req_id}] SCCI TRACE_ERR stage=final_ctx_guard err={type(e).__name__}:{e} model={model_id} node={node_name}")

        # v33.2: resolve model name against node advertised models (alias-first ladder)
        n2 = _nodes.get(node_name) or n
        model_id = resolve_model_name_for_node(req_id, n2, model_id, globals().get('CHAT_MODEL_ID') or globals().get('CHAT_FAST_MODEL_ID') or '')
        body['model'] = model_id
        try:
            log.info(f"[{req_id}] → SCCI CLUSTER sid={req_id} nodes={len(_nodes) if '_nodes' in globals() else 'na'} gpu_busy=0 cpu_busy=0 queue=0")
            log.info(
                f"[{req_id}] → SCCI STATE sid={req_id} ctx={trace_ctx_needed if trace_ctx_needed is not None else 'na'}/"
                f"{trace_phone_ctx if trace_phone_ctx is not None else 'na'} ratio={float(trace_ratio) if trace_ratio is not None else 0.0:.2f} "
                f"node={node_name} prewarm={trace_prewarm} handover={trace_handover} bearer_locked={1 if stream else 0}"
            )
        except Exception:
            pass
        _sc_current = session.get("current_node") or "B-PHONE0"
        _sc_action = 'prewarm' if int(trace_prewarm or 0)==1 and int(trace_handover or 0)==0 else ('handover' if ('handover' in str(fast_reason) or _sc_current != node_name) else 'stay')
        _sc_lane = bucket if 'bucket' in locals() else fast_bucket
        _session_controller_decide(
            session,
            lane=_sc_lane,
            current_node=session.get("current_node") or "B-PHONE0",
            target_node=node_name,
            target_model=model_id,
            action=_sc_action,
            cause=str(fast_reason),
            ctx_needed=int(trace_ctx_needed or 0) if trace_ctx_needed is not None else 0,
            ctx_limit=int(trace_phone_ctx or 0) if trace_phone_ctx is not None else 0,
            ctx_ratio=float(trace_ratio or 0.0) if trace_ratio is not None else 0.0,
            prewarm=int(trace_prewarm or 0),
            handover=int(trace_handover or 0),
        )
        if locals().get("session", None) is not None:
            _session_controller_log(req_id, session)
        log.info(f"[{req_id}] → SCCI DECISION sid={req_id} lane={'tools' if int(body.get('_scci_forced_tool_lane',0) or 0) == 1 else _sc_lane} action={_sc_action} current={session.get('current_node') or 'B-PHONE0'} target={node_name} cause={'tools_internal_ui_task' if int(body.get('_scci_forced_tool_lane',0) or 0) == 1 else fast_reason} model={model_id}")
        _session_controller_commit_route(session, node_name=node_name, model_id=model_id, stream=bool(stream))
        log.info(f"[{req_id}] → ROUTE sid={req_id} node={node_name} model={model_id} bucket={'tools' if int(body.get('_scci_forced_tool_lane',0) or 0) == 1 else ('coding' if int(body.get('_scci_forced_coding_lane',0) or 0) == 1 else ('chat' if int(body.get('_scci_forced_chat_ladder',0) or 0) == 1 else fast_bucket))} reason={'tools_internal_ui_task' if int(body.get('_scci_forced_tool_lane',0) or 0) == 1 else ('coding_intent' if int(body.get('_scci_forced_coding_lane',0) or 0) == 1 else ('longtext_chat_ladder' if int(body.get('_scci_forced_chat_ladder',0) or 0) == 1 else fast_reason))} stream={stream}")
        return await forward(
            req=req,
            body=body,
            req_id=req_id,
            session_id=session_id,
            bucket=fast_bucket,
            node_name=node_name,
            model_id=model_id,
            tried=tried,
            reason=fast_reason,
            started=started,
            extra_headers=None,
        )
    # Otherwise, continue to the generic candidate-selection path below.
    msgs = body.get("messages") or []
    # --- SICC: deep/code detection + ctx escalation with prewarm/handover (model-centric) ---
    # Runs only for router-auto and when no direct node/model is forced.
    if (model_in == "SCCI-Router" or not model_in) and not direct_node:
        all_text = _extract_text_for_estimate(msgs)
        last_user = _last_user_text(msgs)

        est_tokens = _estimate_tokens(all_text)
        tokens_needed = est_tokens + ROUTER_CTX_COMPLETION_RESERVE

        current_model_id = CHAT_MODEL_ID if 'CHAT_MODEL_ID' in globals() else str(body.get("model") or "")
        ctx_current = _parse_ctx_from_model_id(current_model_id) or 2048
        ratio = tokens_needed / max(1, ctx_current)

        is_code_or_deep = _detect_code_or_deep(last_user)
        is_long_paste = len(last_user) >= 2000

        coding_score = 0
        if is_code_or_deep:
            coding_score += 4
        if _CODE_PAT.search(last_user or ""):
            coding_score += 3
        if re.search(r"\b(debug|debugging|understand\s+this\s+code|explain\s+this\s+code|fix\s+this\s+code|help\s+me\s+debug|traceback|exception|fastapi|python|javascript|api|regex|code)\b", last_user or "", re.IGNORECASE):
            coding_score += 3
        log.info(
            f"[{req_id}] → SCCI INTENT sid={req_id} coding={1 if is_code_or_deep else 0} "
            f"large_input={1 if is_long_paste else 0} score={coding_score}"
        )

        target_model = _pick_coder_model(tokens_needed)
        if is_code_or_deep:
            log.info(f"[{req_id}] → SCCI CODER_LADDER sid={req_id} pick={target_model} tokens_needed~{tokens_needed}")
        cand_nodes = [CODER_NODE_PRIMARY, CODER_NODE_FALLBACK]

        # v1.7 hard admission for coding lane: if coding intent is detected and this is not vision,
        # force coder ladder selection before normal chat/direct fallback logic.
        if is_code_or_deep and not bool(_img_n):
            chosen_node = None
            for nn in cand_nodes:
                n = _nodes.get(nn)
                if not n:
                    continue
                if getattr(n, "models", None) is None or target_model in getattr(n, "models", {}):
                    chosen_node = nn
                    log.info(
                        f"[{req_id}] → SCCI SCHED sid={req_id} lane=coding primary={CODER_NODE_PRIMARY} "
                        f"selected={chosen_node} reason={'cpu_primary' if chosen_node == CODER_NODE_PRIMARY else 'cpu_unavailable_gpu_fallback'} "
                        f"model={target_model}"
                    )
                    break
            if chosen_node:
                node_name = chosen_node
                model_id = target_model
                bucket = "coding"
                fast_bucket = "coding"
                fast_reason = "coding_intent"
                trace_ctx_needed = int(tokens_needed or 0)
                trace_phone_ctx = int(_parse_ctx_from_model_id(target_model) or 4096)
                trace_ratio = float(trace_ctx_needed) / float(max(1, trace_phone_ctx))
                trace_prewarm = 0
                trace_handover = 0
                try:
                    session["pinned_node"] = node_name
                    session["pinned_model"] = model_id
                    session["pinned_capability"] = 2
                except Exception:
                    pass

        if (is_code_or_deep or is_long_paste) and ratio >= ROUTER_CTX_PREWARM_RATIO and ratio < ROUTER_CTX_HANDOVER_RATIO:
            for nn in cand_nodes:
                n = _nodes.get(nn)
                if not n:
                    continue
                if getattr(n, "models", None) is None or target_model in getattr(n, "models", {}):
                    asyncio.create_task(_prewarm_node_model(req_id, nn, n, target_model))
                    break

        if is_code_or_deep or (is_long_paste and ratio >= ROUTER_CTX_HANDOVER_RATIO) or (ratio >= ROUTER_CTX_HANDOVER_RATIO):
            chosen_node = None
            chosen_reason = "coding_intent" if is_code_or_deep else "ctx_handover"
            for nn in cand_nodes:
                n = _nodes.get(nn)
                if not n:
                    continue
                if getattr(n, "models", None) is None or target_model in getattr(n, "models", {}):
                    chosen_node = nn
                    if is_code_or_deep:
                        log.info(
                            f"[{req_id}] → SCCI SCHED sid={req_id} lane=coding primary={CODER_NODE_PRIMARY} "
                            f"selected={chosen_node} reason={'cpu_primary' if chosen_node == CODER_NODE_PRIMARY else 'cpu_unavailable_gpu_fallback'} "
                            f"model={target_model}"
                        )
                    break
            if chosen_node:
                body["model"] = target_model
                log.info(
                    f"[{req_id}] ctx_lane_select reason={chosen_reason} est_tokens={est_tokens} "
                    f"reserve={ROUTER_CTX_COMPLETION_RESERVE} ctx_current={ctx_current} ratio={ratio:.2f} "
                    f"to={chosen_node}:{target_model}"
                )
                return await forward(
                    req=req,
                    body=body,
                    req_id=req_id,
                    session_id=session_id,
                    bucket="coding" if is_code_or_deep else "chat",
                    node_name=chosen_node,
                    model_id=target_model,
                    tried=[f"{chosen_node}:{target_model}"],
                    reason=chosen_reason,
                    started=started,
                    extra_headers=None,
                )

    last_user_text = extract_last_user_text(msgs)
    is_task_prompt = is_openwebui_task_prompt(last_user_text)

    # ---- Conditional Time grounding ----
    time_sys = None
    if (not is_task_prompt) and (should_inject_time(last_user_text) or user_requests_past_recall(last_user_text)):
        time_sys = build_time_system_message()

    # ---- Memory retrieval (v8) ----
    mem_sys = None
    lookback = None
    since_ts = None

    if (not is_task_prompt) and MEMORY_ENABLE_V8:
        lookback = parse_lookback_seconds(last_user_text) if user_requests_past_recall(last_user_text) else None
        now_ts = int(time.time())
        since_ts = (now_ts - lookback) if lookback else None

        prof_rows: List[Dict[str, Any]] = []
        proj_rows: List[Dict[str, Any]] = []
        reinforce_ids: List[int] = []

        try:
            prof_rows, prof_ids = await memory_read_v8(
                SCOPE_PROFILE, user_id, None, session_id, since_ts, MEMORY_MAX_PROFILE
            )
            proj_rows, proj_ids = await memory_read_v8(
                SCOPE_PROJECT, None, chat_id, session_id, since_ts, MEMORY_MAX_PROJECT
            )
            reinforce_ids = prof_ids + proj_ids
            mem_sys = build_memory_system_message_v8(prof_rows, proj_rows, include_time=user_requests_past_recall(last_user_text))

            # Reinforce what we actually exposed to the model (keeps “working set” alive)
            if reinforce_ids:
                asyncio.create_task(asyncio.to_thread(db_reinforce_memories_v8, reinforce_ids, MEMORY_RECALL_BOOST))
        except Exception as e:
            log.debug(f"[memory_read_v8] failed: {e!r}")

    # Build final message list: (time?) -> (memory?) -> global prompt (optional) -> user msgs
    msgs2: List[Dict[str, Any]] = []
    if time_sys:
        msgs2.append(time_sys)
    if mem_sys:
        msgs2.append(mem_sys)
    msgs2.extend(msgs)

    msgs2 = apply_global_system_prompt(msgs2)
    body["messages"] = msgs2

    # ---- Memory write (async) from ORIGINAL user text ----
    if (not is_task_prompt) and last_user_text and MEMORY_ENABLE_V8:
        asyncio.create_task(memory_write_task(last_user_text, session_id, user_id, chat_id))

    img_count = count_images_in_last_user_message(msgs2)
    text_tok = approx_text_tokens(msgs2)
    img_tok = img_count * VISION_TOKENS_PER_IMAGE
    prompt_tok_est = text_tok + img_tok
    # Long-prompt pre-routing (first turn safety): if the user's input is large, avoid starting on small-context buckets.
    # Default threshold is 1024 tokens (ROUTER_LONGPROMPT_THRESHOLD_TOKENS).
    longprompt_thr = int(os.getenv("ROUTER_LONGPROMPT_THRESHOLD_TOKENS", "1024"))
    is_longprompt = (text_tok > longprompt_thr)

    max_tokens = get_max_tokens(body)

    bucket = infer_bucket(req, body, last_user_text, img_count, prompt_tok_est, max_tokens)

    # v32.5 SICC-safe: if request contains an image, force vision lane

    if has_image:

        bucket = "vision_fast"


    # v32.5 SICC-safe: vision overrides pinned non-vision model

    try:

        _pm = session.get("pinned_model")

        if bucket.startswith("vision") and _pm and not is_vision_model(str(_pm)):

            log.info(f"[{req_id}] unpin_nonvision_due_to_vision pinned={_pm}")

            session["pinned_model"] = None

            session["pinned_node"] = None

            session["pinned_capability"] = None

            pinned = None

    except Exception:

        pass
    # TOOL_BUCKET_FORCE_UNPIN: tool bucket should never be pinned by client.

    # v34.3: text-only requests must not stick to a pinned vision model
    try:
        _pm = session.get("pinned_model")
        if img_count == 0 and _pm and is_vision_model(str(_pm)):
            log.info(f"[{req_id}] unpin_vision_due_to_text pinned={_pm}")
            session["pinned_model"] = None
            session["pinned_node"] = None
            session["pinned_capability"] = None
            pinned = None
    except Exception:
        pass
    if bucket == "tool":
        pinned = False

    # PIN_GUARD: never follow tool pins for normal buckets
    if bucket == "tool":
        pinned = None
    elif pinned and FUNCTION_MODEL_PAT.search(str(pinned[1])):
        pinned = None

    # TOOL_BUCKET_NO_PIN: tool pre-pass must never follow or update session pin
    if bucket == "tool":
        pinned = None


    # Text-only follow-up reasoning on previous image: inject cached facts.
    if bucket == "vision_detail_text":
        sid = body.get("chat_id") or req.headers.get("x-openwebui-chat-id") or body.get("session") or ""
        facts = get_cached_vision_facts(str(sid)) if sid else ""
        if facts:
            messages.insert(0, {"role": "system", "content": "IMAGE_FACTS_FROM_PREVIOUS_STEP:\n" + facts})

    # If the very first user message is long, escalate bucket immediately.
    if is_longprompt:
        if bucket in ("vision_fast", "vision_reasoning"):
            bucket = "vision_reasoning"
        elif bucket in ("micro", "chat", "deep", "code"):
            bucket = "longctx"

    need_vision = img_count > 0
    stream = bool(body.get("stream", False))
    tried: List[str] = []

    log.info(f"[{req_id}] start bucket={bucket} images={img_count} stream={stream} pinned={bool(pinned)} forced={bool(force_node or force_model)} lookback={lookback}")

    for n in _nodes.values():
        asyncio.create_task(update_node_metrics(n))
        asyncio.create_task(check_node_ready(n))


    # ---- OpenWebUI metadata tasks: always stick to the pinned session model (no fallback) ----
    # OpenWebUI often sends additional "### Task:" prompts (title/tags/follow-ups/etc.).
    # These should NOT trigger capability escalation or route changes; they should follow
    # the already-selected model for the current chat_id/session_id.
    #
    # Exception: requests with images must still route to a vision-capable model.
    if is_task_prompt and pinned and (not FUNCTION_MODEL_PAT.search(str(pinned[1]))) and not (force_node or force_model) and body.get("model") == "SCCI-Router" and img_count == 0:
        pn, pm = pinned
        # Meta-task detour: if the session is pinned to a vision model, don't waste GPU/vision.
        # Route OpenWebUI meta tasks (title/tags/followups) to B-PHONE0 using the normal chat model,
        # without updating the session pin (soft exception).
        detour_from_vision = is_vision_model(pm)
        if detour_from_vision:
            # Meta-task detour (soft): avoid using a vision model for title/tags/followups.
            # Primary: B-PHONE0 (chat model already loaded for the user)
            # Fallback: B-CPU0
            detour_candidates = ["B-PHONE0", "B-CPU0"]
            chosen_detour = None
            chosen_model = None
            for dn in detour_candidates:
                dm = bucket_hint(dn, "chat") or CHAT_MODEL_ID
                dnode = _nodes.get(dn)
                if not dnode:
                    continue
                if dm not in dnode.models:
                    continue
                if not ctx_fits(dn, prompt_tok_est, max_tokens):
                    continue
                ok = dnode.healthy_cached() or await check_node_health(dnode)
                ready = await check_node_ready(dnode)
                if ok and ready:
                    chosen_detour = dn
                    chosen_model = dm
                    break
            if chosen_detour and chosen_model:
                pn = chosen_detour
                pm = chosen_model

        node_for_pin = _nodes.get(pn)
        if node_for_pin and (pm in node_for_pin.models) and ctx_fits(pn, prompt_tok_est, max_tokens):
            ok = node_for_pin.healthy_cached() or await check_node_health(node_for_pin)
            ready = await check_node_ready(node_for_pin)
            if ok and ready:
                tried.append(f"{pn}:{pm} (pinned_meta)")
                return await forward(
                    req=req,
                    body=body,
                    req_id=req_id,
                    session_id=session_id,
                    bucket=bucket,
                    node_name=pn,
                    model_id=pm,
                    tried=tried,
                    reason=("meta_task_detour_from_vision" if detour_from_vision else "session_pinned_meta_task"),
                    started=started,
                )


    # ---------------- v9 Capability routing (stateful + sticky upgrade) ----------------
    # Philosophy (KISS):
    # - "bucket" describes the current message (chat/code/vision/...).
    # - "capability_level" describes the *session* (what model tier we should keep using).
    # - We escalate capability when we see technical signals or long-context pressure.
    # - We only downgrade on explicit reset, long idle, or repeated simple turns.

    now = time.time()
    ctx_ratio_bn10 = float(prompt_tok_est + max_tokens) / float(max(1, node_max_ctx("B-PHONE0")))
    ctx_pressure_bn10 = ctx_ratio_bn10 >= float(CAP_CTX_PRESSURE_RATIO)

    cap_reason = "default"
    if user_explicit_reset(last_user_text):
        cap_reason = "user_reset"
        await reset_session(session_id)
        session = await get_session_state(session_id)
        pinned = _session_pinned_tuple(session)
    else:
        tech = is_technical_signal(last_user_text, bucket)

        # Long context / complex: escalate to level 3
        if bucket == "longctx" or ctx_ratio_bn10 >= float(CAP_LONGCTX_RATIO):
            if int(session.get("capability_level") or 1) < 3:
                cap_reason = "longctx"
            session["capability_level"] = max(int(session.get("capability_level") or 1), 3)
            session["last_tech_ts"] = now
            session["simple_streak"] = 0

        # Engineering: escalate to level 2
        if tech:
            if int(session.get("capability_level") or 1) < 2 and cap_reason == "default":
                cap_reason = "tech_signal"
            session["capability_level"] = max(int(session.get("capability_level") or 1), 2)
            session["last_tech_ts"] = now
            session["simple_streak"] = 0
        else:
            # Conservative downgrade logic:
            # - only count a "simple streak" when the message is short and has no technical markers
            if is_simple_turn(last_user_text):
                session["simple_streak"] = int(session.get("simple_streak") or 0) + 1
            else:
                session["simple_streak"] = 0

            if int(session.get("capability_level") or 1) > 1 and int(session.get("simple_streak") or 0) >= int(CAP_DOWNGRADE_TURNS):
                cap_reason = "cooldown_downgrade"
                session["capability_level"] = 1
                session["pinned_node"] = None
                session["pinned_model"] = None
                session["pinned_capability"] = None
                session["pinned_at"] = 0.0
                pinned = None

        # Keep session fresh
        session["updated_at"] = now

    cap_level = int(max(1, min(3, int(session.get("capability_level") or 1))))

    # Decide the preferred model for this *capability tier*
    preferred_model = CHAT_MODEL_ID
    if cap_level == 2:
        preferred_model = ENGINEERING_MODEL_ID
    elif cap_level == 3:
        preferred_model = LONGCTX_MODEL_ID

    # Determine if we need a vision-capable model for this request
    need_vis = need_vision and bucket in ("vision_fast", "vision_reasoning", "health")

    # GPU is required when:
    # - vision/health buckets (images or medical vision)
    # - code bucket (coding/debugging benefits from GPU model)
    # - optionally deep bucket if forced
    need_gpu = bool(need_vis or bucket in ("code", "health") or (bucket == "deep" and ROUTER_DEEP_FORCE_GPU))

    # Node priority chain:
    # - cap=1: BN10 first unless ctx is near full (then PC-CPU first)
    # - cap=2: PC-GPU first (reasoning/coding), then PC-CPU
    # - cap=3: PC-CPU first (big ctx), then PC-GPU
    chain = capability_node_chain(cap_level, bucket=bucket, ctx_pressure_bn10=ctx_pressure_bn10, need_vision=need_vis, need_gpu=need_gpu)


    # Vision fallback guard: allow CPU0 for vision only when GPU0 is unavailable or busy.
    allow_cpu_vision_fallback = False
    gpu_ok = gpu_ready = True
    gpu_busy = False
    if need_vis:
        gpu = _nodes.get("B-GPU0")
        if gpu:
            gpu_ok = (gpu.healthy_cached() or await check_node_health(gpu))
            gpu_ready = await check_node_ready(gpu)
            # Update metrics (best-effort) so "busy" decision is deterministic.
            try:
                await update_node_metrics(gpu)
            except Exception:
                pass
            gpu_busy = bool(gpu.metrics.loading) or (int(gpu.metrics.queue_depth or 0) >= GPU_BUSY_QUEUE_DEPTH) or (int(gpu.metrics.active_requests or 0) >= GPU_BUSY_ACTIVE_REQS)
            # Stability: never mark GPU busy based solely on latency EMA.
            # Guard against stale metrics / EMA-induced flapping.
            if (not gpu.metrics.loading) and (int(gpu.metrics.queue_depth or 0) == 0) and (int(gpu.metrics.active_requests or 0) == 0):
                gpu_busy = False
        else:
            gpu_ok = False
            gpu_ready = False
        allow_cpu_vision_fallback = (not gpu_ok) or (not gpu_ready) or gpu_busy
        log.info(f"[{req_id}] vision_fallback cpu_allowed={int(allow_cpu_vision_fallback)} gpu_ok={int(gpu_ok)} gpu_ready={int(gpu_ready)} gpu_busy={int(gpu_busy)} q={int(gpu.metrics.queue_depth) if gpu else -1} act={int(gpu.metrics.active_requests) if gpu else -1} loading={int(gpu.metrics.loading) if gpu else -1}")

    # 1) Pinned reuse (sticky *session* routing):
    # Reuse only if:
    # - not forced
    # - still same capability tier (prevents pinning to "weak" model after upgrade)
    # - node is healthy + ready
    # - model exists + context fits
    if pinned and not (force_node or force_model) and body.get("model") == "SCCI-Router":
        pn, pm = pinned
        pinned_cap = int(session.get("pinned_capability") or 0)

        # IMPORTANT (v9 KISS preference):
        # For capability level 1 (light chat), BN10 should be preferred whenever possible.
        # So we only reuse a level-1 pin if it matches the current top-choice node in the chain.
        # This prevents a previous temporary failover (e.g. BN10 offline) from keeping the session stuck on PC-CPU.
        if cap_level == 1 and chain:
            if pn != chain[0]:
                pinned_cap = -1  # force bypass of pin reuse

        # Hard invariants: pinned reuse must not violate modality.
        # - If images are present, the model MUST be vision-capable.
        # - If no images, avoid sticking to a vision model for normal text buckets.
        if pinned_cap == cap_level:
            node_for_pin = _nodes.get(pn)
            if not node_for_pin or (pm not in node_for_pin.models):
                pinned_cap = -1
            else:
                is_pin_vision = bool(node_for_pin.models[pm].vision)
                if need_vis and not is_pin_vision:
                    pinned_cap = -1
                if need_vis and pn == "B-CPU0" and not allow_cpu_vision_fallback:
                    pinned_cap = -1
                if (not need_vis) and is_pin_vision and bucket not in ("vision_fast", "vision_reasoning", "health"):
                    pinned_cap = -1
        
                if pinned_cap == cap_level and ctx_fits(pn, prompt_tok_est, max_tokens):
                    node = _nodes.get(pn)
                    if node:
                        ok = node.healthy_cached() or await check_node_health(node)
                        ready = await check_node_ready(node)
                        if ok and ready and (pm in node.models):
                            tried.append(f"{pn}:{pm} (pinned_cap{cap_level})")
                            extra = {
                                "X-Capability-Level": str(cap_level),
                                "X-Capability-Reason": cap_reason,
                                "X-Capability-SimpleStreak": str(int(session.get("simple_streak") or 0)),
                                "X-CtxRatio-BN10": f"{ctx_ratio_bn10:.3f}",
                            }
                            return await forward(
                                req, body, req_id, session_id, bucket, pn, pm, tried,
                                reason="session_pinned", started=started, extra_headers=extra
                            )

    # 2) Capability-aware routing
    #
    # IMPORTANT POLICY (your Tier-1 philosophy):
    # - cap_level=1 is "Phone-first" to save PC CPU/RAM.
    #   We pick the FIRST node in the chain that can serve the request (strict order),
    #   and we only fall back to scoring if everything in the chain is blocked.
    # - cap_level>=2 keeps score-based selection (performance matters more there).

    # --- Tier-1 strict preference (cap=1): phone-first, then CPU, then GPU ---
    # We do this to avoid unnecessary PC CPU/RAM usage when BN10 can handle the request.
    if cap_level == 1 and not (force_node or force_model) and body.get("model") == "SCCI-Router" and not need_vis:
        for node_name in chain:
            node = _nodes.get(node_name)
            if not node:
                continue

            # Vision buckets: CPU0 is allowed only as a fallback when GPU0 is unavailable/busy.
            if need_vis and node_name == "B-CPU0" and not allow_cpu_vision_fallback:
                tried.append(f"{node_name} (vision_cpu_blocked)")
                continue

            # Health / readiness gates
            if not (node.healthy_cached() or await check_node_health(node)):
                tried.append(f"{node_name} (unhealthy:{node.status})")
                continue
            if not await check_node_ready(node):
                tried.append(f"{node_name} (not_ready:{node.ready_status})")
                continue
            if node.metrics_fresh() and node.metrics.loading:
                tried.append(f"{node_name} (loading)")
                continue
            if not ctx_fits(node_name, prompt_tok_est, max_tokens):
                tried.append(f"{node_name} (ctx_overflow)")
                continue

            # Coherence: prefer the same chat model id across tiers/nodes when possible.
            chosen: Optional[str] = None
            if preferred_model and preferred_model in node.models and ((not need_vis) or node.models[preferred_model].vision):
                chosen = preferred_model
            else:
                hint = bucket_hint(node_name, bucket)
                hinted = find_model_by_hint(node, hint, bucket) if hint else None
                chosen = hinted or pick_best_for_bucket(node, bucket, need_vision=need_vis)

            if not chosen:
                tried.append(f"{node_name} (no_model)")
                continue

            # CPU vision fallback: restrict to a small @1k vision model when GPU0 is unavailable/busy.
            if need_vis and node_name == "B-CPU0" and allow_cpu_vision_fallback:
                cpu_pick = None
                if VISION_CPU_FALLBACK_MODEL_ID in node.models:
                    cpu_pick = VISION_CPU_FALLBACK_MODEL_ID
                else:
                    # Best-effort: pick any vision model with '@1k' in id.
                    for mid in node.models.keys():
                        if node.models[mid].vision and "@1k" in str(mid):
                            cpu_pick = mid
                            break
                if cpu_pick:
                    chosen = cpu_pick
                    cap_reason = (cap_reason + "|cpu_vis_fallback")
                else:
                    # Stability: do NOT route vision to CPU0 unless a small @1k vision model exists locally.
                    tried.append(f"{node_name} (no_1k_vision_fallback)")
                    continue


            # Persist session pin (sticky routing)


            PIN_WRITE_BUCKETS = {"chat","deep","vision_fast","vision_reasoning","vision_detail_text","health"}
            if pinned and (bucket in PIN_WRITE_BUCKETS) and (not FUNCTION_MODEL_PAT.search(str(chosen))):


                session["pinned_node"] = node_name


                session["pinned_model"] = chosen


                session["pinned_capability"] = cap_level


                session["pinned_at"] = now

            # Update per-node stickiness (helps llama.cpp router-mode avoid thrash)
            record_bucket(node, bucket)
            set_sticky_model(node, bucket, chosen)

            # Tool bucket: obey bucket hint strictly (prefer function tool model)

            if bucket == "tool":

                hinted_tool = bucket_hint(node_name, bucket)

                if hinted_tool:

                    chosen = hinted_tool

                    cap_reason = "tool_hint"


            tried.append(f"{node_name}:{chosen} (policy_cap1)")
            extra = {
                "X-Capability-Level": str(cap_level),
                "X-Capability-Reason": cap_reason,
                "X-Capability-SimpleStreak": str(int(session.get("simple_streak") or 0)),
                "X-CtxRatio-BN10": f"{ctx_ratio_bn10:.3f}",
                "X-Capability-PreferredModel": preferred_model,
            }
            return await forward(
                req, body, req_id, session_id, bucket, node_name, chosen, tried,
                reason="capability_level_1_policy", started=started, extra_headers=extra
            )

    # --- cap>=2 or policy-cap1 chain blocked: score-based within the chain ---
    candidates: List[Tuple[float, str, str, str]] = []

    for node_name in chain:
        node = _nodes.get(node_name)
        if not node:
            continue

        # Vision buckets: CPU0 is allowed only as a fallback when GPU0 is unavailable/busy.
        if need_vis and node_name == "B-CPU0" and not allow_cpu_vision_fallback:
            tried.append(f"{node_name} (vision_cpu_blocked)")
            continue

        # Optional force-node header support
        if force_node and node_name != force_node:
            continue

        # Health / readiness gates (battery/offline => failover automatically)
        if not (node.healthy_cached() or await check_node_health(node)):
            tried.append(f"{node_name} (unhealthy:{node.status})")
            continue

        if not await check_node_ready(node):
            tried.append(f"{node_name} (not_ready:{node.ready_status})")
            continue

        if node.metrics_fresh() and node.metrics.loading:
            tried.append(f"{node_name} (loading)")
            continue

        if not ctx_fits(node_name, prompt_tok_est, max_tokens):
            tried.append(f"{node_name} (ctx_overflow)")
            continue

        chosen: Optional[str] = None
        sticky_hit = False
        
        def _model_ok(mid: str) -> bool:
            # Enforce modality invariants:
            # - If vision is required, only vision-capable models are allowed.
            # - If vision is not required, we prefer non-vision models for normal text buckets.
            mi = node.models.get(mid)
            if not mi:
                return False
            if need_vis and not mi.vision:
                return False
            if (not need_vis) and mi.vision and bucket not in ("vision_fast", "vision_reasoning", "health"):
                return False
            return True
        
        # Forced model overrides everything (but still must exist on the node).
        if force_model:
            if force_model in node.models:
                # If the user forces a model, we honor it even if it's a vision model for text;
                # but if images are present, we still require a vision-capable model.
                if need_vis and not node.models[force_model].vision:
                    tried.append(f"{node_name} (forced_model_not_vision)")
                    continue
                chosen = force_model
            else:
                tried.append(f"{node_name} (forced_model_missing)")
                continue
        else:
            # Build a small ordered preference list per node/bucket (CPU-first philosophy).
            desired: List[str] = []
        
            if need_vis:
                # v34: enforce vision ctx-ladder pick if present (prevents downgrade to @1k)
                if vision_forced_model:
                    desired = [vision_forced_model, VISION_4B_8K, VISION_4B_4K, VISION_4B_2K, VISION_FAST_MODEL_ID]
                else:
                    desired = [VISION_4B_8K, VISION_4B_4K, VISION_4B_2K, VISION_FAST_MODEL_ID, VISION_MODEL_ID]
            elif bucket in ("chat", "micro", "longctx"):
                desired = [CHAT_MODEL_ID]
            elif bucket == "deep":
                if node_name == "B-CPU0":
                    desired = [DEEP_CPU_MODEL_ID, CHAT_MODEL_ID]
                elif node_name == "B-GPU0":
                    desired = [DEEP_GPU_MODEL_ID, ENGINEERING_MODEL_ID, DEEP_CPU_MODEL_ID, CHAT_MODEL_ID]
                else:
                    desired = [CHAT_MODEL_ID]
            elif bucket == "code":
                if node_name == "B-GPU0":
                    desired = [ENGINEERING_MODEL_ID, DEEP_GPU_MODEL_ID, DEEP_CPU_MODEL_ID]
                else:
                    desired = [ENGINEERING_MODEL_ID, DEEP_CPU_MODEL_ID, CHAT_MODEL_ID]
            elif bucket == "health":
                # Prefer vision model if available; otherwise fall back to engineering/deep.
                desired = [VISION_MODEL_ID, ENGINEERING_MODEL_ID, DEEP_CPU_MODEL_ID, CHAT_MODEL_ID]
            else:
                desired = [CHAT_MODEL_ID, preferred_model]
        
            # Prefer the capability-tier model only if it passes modality and fits the desired list.
            if preferred_model and preferred_model in node.models and _model_ok(preferred_model) and (preferred_model in desired):
                chosen = preferred_model
            else:
                # Try desired list in order
                for mid in desired:
                    # Function/tool models must not serve non-tool buckets
                    if ("bucket" in locals() and bucket != "tool") and FUNCTION_MODEL_PAT.search(str(mid)):
                        continue
                    if mid and (mid in node.models) and _model_ok(mid):
                        chosen = mid
                        break
        
            # Otherwise keep existing v8 behavior: sticky-by-bucket then hints then best score for bucket
            if not chosen:
                sticky = get_sticky_model(node, bucket)
                if sticky and sticky in node.models and _model_ok(sticky):
                    chosen = sticky
                    sticky_hit = True
                else:
                    hint = bucket_hint(node_name, bucket)
                    hinted = find_model_by_hint(node, hint, bucket) if hint else None
                    if hinted and _model_ok(hinted):
                        chosen = hinted
                    else:
                        chosen = pick_best_for_bucket(node, bucket, need_vision=need_vis)
        
            # Final safety check (belt + suspenders)
            if chosen and not _model_ok(chosen):
                chosen = None
            
            if not chosen:
                tried.append(f"{node_name} (no_model)")
                continue
            
            switch_allowed = True
            if should_avoid_switch(node) and not bucket_switch_allowed(node, bucket):
                switch_allowed = False
        
            score = score_node_for_request(node, bucket, sticky_hit=sticky_hit, switch_allowed=switch_allowed)
            candidates.append((score, node_name, chosen, "cap" if chosen == preferred_model else ("sticky" if sticky_hit else "pick")))
        
    if not candidates:
        # No node could serve the request (all unhealthy, not ready, ctx overflow, etc.)
        extra = {
            "X-Capability-Level": str(cap_level),
            "X-Capability-Reason": cap_reason,
            "X-Capability-SimpleStreak": str(int(session.get("simple_streak") or 0)),
            "X-CtxRatio-BN10": f"{ctx_ratio_bn10:.3f}",
        }
        log.warning(f"[{req_id}] SCCI DISCOVERY_RETRY cause=no_candidates")
        try:
            await discover_nodes()
            await asyncio.sleep(0.15)
        except Exception as _disc_retry_err:
            log.warning(f"[{req_id}] SCCI discovery_retry_err={type(_disc_retry_err).__name__}:{_disc_retry_err}")

        candidates = select_candidates()
        if candidates:
            log.info(f"[{req_id}] SCCI RETRY_SUCCESS node_count={len(candidates)}")
        else:
            resp = JSONResponse({"error": {"message": "No healthy nodes/models available", "type": "router_no_candidates"}}, status_code=503)
            set_headers(resp, req_id, session_id, bucket, "", "", "no_candidates", tried, extra=extra)
            return resp

    candidates.sort(key=lambda x: x[0])
    score, node_name, model_id, how = candidates[0]
    tried.append(f"{node_name}:{model_id} ({how})")

    # CPU vision fallback: restrict to a small @1k vision model when GPU0 is unavailable/busy.
    if need_vis and node_name == "B-CPU0" and allow_cpu_vision_fallback:
        cpu_pick = None
        if VISION_CPU_FALLBACK_MODEL_ID and node and (VISION_CPU_FALLBACK_MODEL_ID in node.models):
            cpu_pick = VISION_CPU_FALLBACK_MODEL_ID
        else:
            for mid in (node.models.keys() if node else []):
                if node.models[mid].vision and "@1k" in str(mid):
                    cpu_pick = mid
                    break
        if cpu_pick:
            model_id = cpu_pick
            tried.append(f"{node_name}:{model_id} (cpu_vis_fallback)")


    # Persist session pin (sticky routing)
    session["pinned_node"] = node_name
    session["pinned_model"] = model_id
    session["pinned_capability"] = cap_level
    session["pinned_at"] = now

    # Update per-node stickiness (helps llama.cpp router-mode avoid thrash)
    sel_node = _nodes.get(node_name)
    if sel_node:
        record_bucket(sel_node, bucket)
        set_sticky_model(sel_node, bucket, model_id)

    # --- Explain why the *top preferred node* was skipped (debugging aid) ---
    # This helps you answer questions like: "Why didn't it pick BN10 for chat?"
    # We keep it KISS: we reuse the existing `tried[]` reasons collected above.
    if chain:
        top = chain[0]
        if node_name != top:
            # Collect only entries related to the preferred node.
            # Examples that may appear in `tried`:
            #   "bn10 (unhealthy:cooldown)"
            #   "bn10 (not_ready:timeout)"
            #   "bn10 (loading)"
            #   "bn10 (ctx_overflow)"
            #   "bn10 (no_model)"
            #   "bn10:Qwen... (pinned_cap1)"  (rare here)
            top_reasons = [t for t in tried if t.startswith(top + " ") or t.startswith(top + ":")]
            if top_reasons:
                log.info(
                    f"[{req_id}] preferred_node_skipped top={top} reasons={' | '.join(top_reasons)} "
                    f"selected={node_name}:{model_id} cap={cap_level} bucket={bucket} ctx_ratio_bn10={ctx_ratio_bn10:.3f}"
                )

    # Remember sticky model for bucket (v8 behavior) + record bucket switches (router-mode protection)
    node = _nodes.get(node_name)
    if node:
        set_sticky_model(node, bucket, model_id)
        record_bucket(node, bucket)

    # Update the session pin to make the capability tier "sticky"
    await update_session_pin(session_id, session, node_name, model_id)

    extra = {
        "X-Capability-Level": str(cap_level),
        "X-Capability-Reason": cap_reason,
        "X-Capability-SimpleStreak": str(int(session.get("simple_streak") or 0)),
        "X-CtxRatio-BN10": f"{ctx_ratio_bn10:.3f}",
        "X-Capability-PreferredModel": preferred_model,
    }

    # Debug headers: expose routing decision for this response (optional)
    if ROUTER_DEBUG_HEADERS:
        try:
            req.state.router_bucket = bucket
            req.state.router_node = node_name
            req.state.router_model = model_id
            req.state.router_reason = f"capability_level_{cap_level}"
            req.state.router_capability_level = cap_level
            req.state.router_pinned = bool(pinned)
        except Exception:
            pass

    return await forward(
        req, body, req_id, session_id, bucket, node_name, model_id, tried,
        reason=f"capability_level_{cap_level}", started=started, extra_headers=extra
    )

# ---------------- Forwarding + latency EMA update ----------------
async def forward(
    req: Request,
    body: Dict[str, Any],
    req_id: str,
    session_id: str,
    bucket: str,
    node_name: str,
    model_id: str,
    tried: List[str],
    reason: str,
    started: float,
    extra_headers: Optional[Dict[str, str]] = None,
):

    # ===== FIX v6: ensure max_tokens always defined =====
    try:
        if 'max_tokens' not in locals():
            max_tokens = 0
    except Exception:
        max_tokens = 0
    # ====================================================


    # ===== FIX v5: ensure prompt_tok_est always defined =====
    try:
        text_blob = _extract_text_for_estimate(messages)
        prompt_tok_est = _estimate_tokens(text_blob)
    except Exception:
        prompt_tok_est = 0
    if not isinstance(prompt_tok_est, int):
        prompt_tok_est = 0
    # =======================================================

    # v32.5 SICC-safe: never send image parts to non-vision models (vision is a tool lane)
    if not is_vision_model(str(model_id or "")):
        try:
            if isinstance(body, dict) and isinstance(body.get("messages"), list):
                new_msgs = []
                for m in body["messages"]:
                    if isinstance(m, dict) and isinstance(m.get("content"), list):
                        txt = []
                        for p in m.get("content", []):
                            if isinstance(p, dict) and p.get("type") == "text":
                                t = p.get("text", "")
                                if t:
                                    txt.append(str(t))
                        m2 = dict(m)
                        m2["content"] = "\n".join(txt).strip()
                        new_msgs.append(m2)
                    else:
                        new_msgs.append(m)
                body["messages"] = new_msgs
        except Exception:
            pass

    # v32.3: vision-as-tool safeguard — never send image parts to non-vision models
    def _strip_images_from_messages(_msgs):
        out = []
        for mm in (_msgs or []):
            if not isinstance(mm, dict):
                continue
            c = mm.get("content")
            if isinstance(c, list):
                txt_parts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if t:
                            txt_parts.append(str(t))
                mm2 = dict(mm)
                mm2["content"] = "\n".join(txt_parts).strip()
                out.append(mm2)
            else:
                out.append(mm)
        return out
    """Forward one OpenAI-compatible request to a selected node/model.

    v28 fixes:
    - Always sanitize multimodal message parts when targeting text-only models (no mmproj).
    - Tool pre-pass: retry once on B-CPU0, then degrade gracefully (200 with empty tool_calls) BUT log loudly.
    - Log downstream 4xx/5xx bodies (truncated) for debugging.
    - Attach debug headers via request.state (middleware will emit).
    """
    node = _nodes[node_name]
    # v32.3: sanitize body for text-only targets (remove image_url parts)

    try:

        _mid = str(model_id or "")

        if _mid and (not is_vision_model(_mid)):

            if isinstance(body, dict):

                body = dict(body)

                _msgs = body.get("messages") or []

                body["messages"] = _strip_images_from_messages(_msgs)

    except Exception:

        pass



    _medical_current_turn_image = _current_turn_has_image_payload(body.get("messages") or [])
    _medical_current_text = _last_user_text(_current_turn_only_messages(body.get("messages") or []))
    _medical_demo_active = bool(_medical_current_turn_image and _is_medical_intent_text(_medical_current_text or ""))
    if ROUTER_UI_HELPER_ISOLATE_MEDICAL and str(bucket or "") == "ui_helper":
        _medical_demo_active = False
    log.info(
        f"[{req_id}] → SCCI SAFETY_SCOPE sid={req_id} medical_active={1 if _medical_demo_active else 0} "
        f"current_turn_image={1 if _medical_current_turn_image else 0} lane={bucket}"
    )

    # deterministic medical vision routing (current-turn scoped only)
    try:
        if _medical_demo_active:
            node_name = MEDGEMMA_DEFAULT_NODE
            model_id = MEDGEMMA_DEFAULT_MODEL
            log.info(f"[{req_id}] → SCCI MEDICAL_ROUTE sid={req_id} node={MEDGEMMA_DEFAULT_NODE} model={MEDGEMMA_DEFAULT_MODEL}")
    except Exception:
        pass

    if _medical_demo_active:
        try:
            log.info(f"[{req_id}] → SCCI MEDICAL_SAFETY sid={req_id} medical_intent=1 bucket={bucket} node={node_name} model={model_id}")
        except Exception:
            pass

    payload = dict(body)
    payload.pop("_router_medical_demo", None)
    _scci_lane_label = str(payload.pop("_scci_lane_label", "") or "").strip()
    _scci_longtext_tool = bool(payload.pop("_scci_longtext_tool", 0))
    _scci_longtext_tool_capture = bool(payload.pop("_scci_longtext_tool_capture", 0))
    _scci_longtext_tool_doc_hash = str(payload.pop("_scci_longtext_tool_doc_hash", "") or "").strip()
    payload["model"] = model_id
    stream = bool(payload.get("stream", False))

    session = await get_session_state(session_id)
    _scci_reset_text = _last_user_text(payload.get("messages") or [])
    _scci_preserve_longtext = bool(
        str(session.get("scci_last_lane") or "").strip() == "longtext_tool"
        and str(session.get("longtext_tool_summary") or "").strip()
        and (
            _detect_openwebui_internal_task(_scci_reset_text)
            or (int(LONGTEXT_TOOL_ACK_BYPASS or 0) == 1 and _is_ack_only_turn(_scci_reset_text))
        )
    )
    _session_isolation_reset(
        session,
        req_id,
        (_scci_lane_label or str(bucket or "chat")),
        preserve_longtext_tool=_scci_preserve_longtext,
    )

    if _medical_demo_active:
        try:
            _msgs = list(payload.get("messages") or [])
            _need_insert = True
            if _msgs and isinstance(_msgs[0], dict) and str(_msgs[0].get("role") or "") == "system":
                _c0 = str(_msgs[0].get("content") or "")
                if "Medical Demo Notice:" in _c0:
                    _need_insert = False
            if _need_insert:
                _msgs.insert(0, {"role": "system", "content": SCCI_MEDICAL_DEMO_DISCLAIMER_IN})
                payload["messages"] = _msgs
            log.info(f"[{req_id}] → SCCI MEDICAL_DISCLAIMER_APPLIED sid={req_id} phase=input applied={1 if _need_insert else 0}")
        except Exception as _medical_in_err:
            log.warning(f"[{req_id}] medical_disclaimer_input_err={type(_medical_in_err).__name__}:{_medical_in_err}")

    # Text-only safety: sanitize only pure text payloads; preserve multimodal/file/audio payloads unchanged.
    try:
        _payload_messages = list(payload.get("messages") or [])
        if not is_vision_model(model_id):
            payload["messages"] = sanitize_for_model(model_id, _payload_messages)
        elif _messages_have_structured_nontext_payload(_payload_messages):
            payload["messages"] = _payload_messages
    except Exception:
        pass

    # Expose routing decision for debug headers (middleware)
    if ROUTER_DEBUG_HEADERS:
        try:
            req.state.router_bucket = bucket
            req.state.router_node = node_name
            req.state.router_model = model_id
            req.state.router_reason = reason
        except Exception:
            pass
# removed legacy route log
    # Bearer lock: once streaming starts, never handover mid-generation.
    if stream:
        if locals().get("session", None) is not None:
            _session_controller_prepare(session, session_id, bucket)
        if locals().get("session", None) is not None:
            _session_controller_commit_route(session, node_name=node_name, model_id=model_id, stream=True)
        log.info(f"[{req_id}] → SCCI BEARER_LOCK sid={req_id} cause=streaming node={node_name} model={model_id} bucket={bucket}")

    headers = passthrough_headers(req)
    if extra_headers:
        headers.update(extra_headers)

    # ---- Streaming path ----
    if stream:
        try:
            upstream = await send_upstream_stream(node.url(OPENAI_CHAT_PATH), headers, payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            _mark_node_transport_down(node, req_id, e)
            fb_node_name, fb_node, fb_mid = await _pick_transport_fallback(req_id, bucket, node_name, model_id, tried, prompt_tok_est=prompt_tok_est if isinstance(prompt_tok_est, int) else 0, max_tokens=max_tokens if isinstance(max_tokens, int) else 0)
            if not fb_node or not fb_mid:
                raise
            tried.append(f"{fb_node_name}:{fb_mid}(transport_fallback:{type(e).__name__})")
            log.warning(f"[{req_id}] STREAM_HANDOVER from={node_name} to={fb_node_name} cause={type(e).__name__} bucket={bucket}")
            upstream = await send_upstream_stream(fb_node.url(OPENAI_CHAT_PATH), headers, {**payload, 'model': fb_mid})
            node_name, model_id, node = fb_node_name, fb_mid, fb_node
            if ROUTER_DEBUG_HEADERS:
                try:
                    req.state.router_node = node_name
                    req.state.router_model = model_id
                    req.state.router_reason = f"{reason}|transport_fallback"
                except Exception:
                    pass

        async def event_iter():
            _scci_stream_chunks = []
            try:
                _medical_disclaimer_sent = False
                async for chunk in upstream.aiter_raw():
                    if _scci_longtext_tool_capture:
                        try:
                            _scci_stream_chunks.append(bytes(chunk))
                        except Exception:
                            pass
                    if _medical_demo_active and (not _medical_disclaimer_sent):
                        _medical_disclaimer_sent = True
                        yield _medical_disclaimer_sse_chunk(req_id, model_id)
                    yield chunk
            finally:
                await upstream.aclose()
                if _scci_longtext_tool_capture:
                    try:
                        _scci_summary_text = _extract_assistant_text_from_sse_chunks(
                            _scci_stream_chunks,
                            max_chars=int(LONGTEXT_TOOL_SUMMARY_MAX_CHARS or 12000),
                        )
                        if _scci_summary_text:
                            _store_longtext_tool_summary(session, _scci_summary_text, _scci_longtext_tool_doc_hash, model_id)
                            log.info(f"[{req_id}] → SCCI LONGTEXT_TOOL sid={req_id} phase=capture mode=stream summary_chars={len(_scci_summary_text)}")
                    except Exception as _lt_stream_cap_err:
                        log.warning(f"[{req_id}] SCCI longtext_tool_stream_capture_err={type(_lt_stream_cap_err).__name__}:{_lt_stream_cap_err}")
                dt_ms = (time.time() - started) * 1000.0
                _update_latency_ema(node, bucket, dt_ms)
                if locals().get("session", None) is not None:
                    _session_controller_complete(session)
                log.info(f"[{req_id}] → SCCI PROC_END sid={req_id} result=ok latency_ms={dt_ms:.0f} mode=stream")
                log.info(f"[{req_id}] → SCCI TIMELINE sid={req_id} state=1ms decision=1ms route=2ms lock=0ms exec={dt_ms:.0f}ms total={dt_ms:.0f}ms")
                log.info(f"[{req_id}] stream_closed dt_ms={dt_ms:.0f}")

        resp = StreamingResponse(event_iter(), media_type=upstream.headers.get("content-type", "text/event-stream"))
        set_headers(resp, req_id, session_id, bucket, node_name, model_id, reason, tried, extra_headers)
        return resp

    # ---- Non-streaming path ----
    async def _post(n: Node, mid: str):
        p = dict(payload)
        p["model"] = mid
        return await client.post(n.url(OPENAI_CHAT_PATH), headers=headers, json=p)

    try:
        r = await _post(node, model_id)
    except httpx.RemoteProtocolError as e:
        log.warning(f"[{req_id}] remote_protocol_error node={node_name} model={model_id} retry_same_node=1 err={e}")
        try:
            retry_timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
            async with httpx.AsyncClient(timeout=retry_timeout, headers={**headers, "Connection": "close"}) as c2:
                p2 = dict(payload)
                p2["model"] = model_id
                r = await c2.post(node.url(OPENAI_CHAT_PATH), headers={**headers, "Connection": "close"}, json=p2)
        except Exception as e2:
            _mark_node_transport_down(node, req_id, e2)
            fb_node_name, fb, fb_mid = await _pick_transport_fallback(req_id, bucket, node_name, model_id, tried, prompt_tok_est=prompt_tok_est if isinstance(prompt_tok_est, int) else 0, max_tokens=max_tokens if isinstance(max_tokens, int) else 0)
            if fb and fb_mid:
                tried.append(f"{fb_node_name}:{fb_mid}(handover_transport_fail:{type(e2).__name__})")
                log.warning(f"[{req_id}] handover node_down from={node_name} to={fb_node_name} model={fb_mid}")
                r = await _post(fb, fb_mid)
                node_name, model_id, node = fb_node_name, fb_mid, fb
            else:
                log.error(f"[{req_id}] node_down_and_no_fallback from={node_name} tried={tried}")
                raise
    except (httpx.ReadTimeout, httpx.ConnectError, httpx.ConnectTimeout) as e:
        _mark_node_transport_down(node, req_id, e)

        if bucket == "tool" and node_name != "B-CPU0":
            fb = _nodes.get("B-CPU0")
            fb_mid = _resolve_fallback_model(req_id, fb, "tool", model_id) if fb else None
            if fb and fb_mid:
                tried.append(f"B-CPU0:{fb_mid}(tool_exc_fallback:{type(e).__name__})")
                try:
                    r = await _post(fb, fb_mid)
                    node_name, model_id, node = "B-CPU0", fb_mid, fb
                except Exception as e2:
                    log.error(f"[tool_fallback_exc] tried={tried} exc={e2}")
                    r = None
            else:
                r = None
        else:
            fb_node_name, fb, fb_mid = await _pick_transport_fallback(req_id, bucket, node_name, model_id, tried, prompt_tok_est=prompt_tok_est if isinstance(prompt_tok_est, int) else 0, max_tokens=max_tokens if isinstance(max_tokens, int) else 0)
            if fb and fb_mid:
                tried.append(f"{fb_node_name}:{fb_mid}(transport_fallback:{type(e).__name__})")
                log.warning(f"[{req_id}] TRANSPORT_HANDOVER from={node_name} to={fb_node_name} cause={type(e).__name__} bucket={bucket}")
                r = await _post(fb, fb_mid)
                node_name, model_id, node = fb_node_name, fb_mid, fb
            else:
                raise

    # If request totally failed and it's tool: degrade gracefully but log
    if bucket == "tool" and r is None:
        log.error(f"[tool_degraded] node_down tried={tried}")
        return JSONResponse(status_code=200, content={"tool_calls": []})

    # Tool bucket: retry CPU fallback once on HTTP error codes too
    if bucket == "tool" and r is not None and r.status_code >= 400 and node_name != "B-CPU0":
        try:
            raw = await r.aread()
            snippet = raw[:2000].decode(errors="replace")
            log.error(f"[tool_primary_err] status={r.status_code} node={node_name} model={model_id} tried={tried} body={snippet}")
        except Exception:
            log.error(f"[tool_primary_err] status={r.status_code} node={node_name} model={model_id} tried={tried}")

        fb = _nodes.get("B-CPU0")
        fb_mid = bucket_hint("B-CPU0", "tool") or model_id
        if fb and fb_mid in fb.models:
            tried.append(f"B-CPU0:{fb_mid}(tool_http_fallback)")
            r = await _post(fb, fb_mid)
            node_name, model_id, node = "B-CPU0", fb_mid, fb

    # If tool still failing, degrade gracefully but log loudly
    if bucket == "tool" and r is not None and r.status_code >= 400:
        try:
            raw = await r.aread()
            snippet = raw[:2000].decode(errors="replace")
            log.error(f"[tool_degraded] status={r.status_code} node={node_name} model={model_id} tried={tried} body={snippet}")
        except Exception:
            log.error(f"[tool_degraded] status={r.status_code} node={node_name} model={model_id} tried={tried}")
        return JSONResponse(status_code=200, content={"tool_calls": []})

    # Non-tool: log backend error bodies (truncated) for diagnosis, but pass through status.
    if bucket != "tool" and r is not None and r.status_code >= 400:
        # Reactive SRVCC: if the phone hits backend ctx overflow, handover to CPU0 text lane once.
        try:
            raw = await r.aread()
            snippet = raw[:4000].decode(errors="replace")
        except Exception as _e:
            log.error(f"[backend_err] req_id={req_id} bucket={bucket} node={node_name} model={model_id} status={r.status_code} (failed to read body: {_e})")
            raw = await r.aread()
            snippet = raw[:4000].decode(errors="replace")

        # Detect llama.cpp ctx overflow (OpenAI-compatible error payload)
        is_ctx_overflow = False
        n_prompt = None
        n_ctx = None
        try:
            j = json.loads(snippet) if snippet and snippet.lstrip().startswith("{") else None
            if isinstance(j, dict):
                err = j.get("error") if isinstance(j.get("error"), dict) else {}
                msg = str(err.get("message", "") or "")
                typ = str(err.get("type", "") or "")
                if "exceed_context_size" in typ or "exceed_context_size" in msg or "exceeds the available context size" in msg:
                    is_ctx_overflow = True
                n_prompt = err.get("n_prompt_tokens") or j.get("n_prompt_tokens")
                n_ctx = err.get("n_ctx") or j.get("n_ctx")
        except Exception:
            # fall back to substring check
            if "exceed_context_size_error" in snippet or "exceeds the available context size" in snippet:
                is_ctx_overflow = True

        # Only for text lanes (chat/direct/micro) and only when originating from B-PHONE0
        if is_ctx_overflow:
            try:
                log.error(f"[{req_id}] backend_ctx_overflow_visible node={node_name} model={model_id} bucket={bucket} detail={snippet}")
            except Exception:
                pass

        if (
            is_ctx_overflow
            and (not body.get("stream", False))
            and (not is_vision_model(str(model_id or "")))
            and str(node_name) == "B-PHONE0"
            and str(bucket) in ("chat", "micro", "direct")
            and (not any(str(t).startswith("B-CPU0:") for t in (tried or [])))
        ):
            target_node = "B-CPU0"
            target_model = DEEP_CPU_MODEL_ID
            # Ensure the chosen CPU fallback model actually exists on CPU0; otherwise pick best text model for deep/chat.
            try:
                cpu_node_obj = _nodes.get("B-CPU0")
            except Exception:
                cpu_node_obj = None
            if cpu_node_obj and getattr(cpu_node_obj, "models", None):
                if target_model not in cpu_node_obj.models:
                    alt = pick_best_for_bucket(cpu_node_obj, "deep", need_vision=False) or pick_best_for_bucket(cpu_node_obj, "chat", need_vision=False)
                    if alt:
                        target_model = alt

            try:
                log.info(
                    f"[{req_id}] HANDOVER chat B-PHONE0->B-CPU0 cause=backend_ctx_overflow "
                    f"prompt_tokens={n_prompt} n_ctx={n_ctx} from={node_name}:{model_id} to={target_node}:{target_model}"
                )
            except Exception:
                log.info(f"[{req_id}] HANDOVER chat B-PHONE0->B-CPU0 cause=backend_ctx_overflow from={node_name}:{model_id} to={target_node}:{target_model}")

            # Route the same request to CPU0 with a deeper text model.
            body2 = dict(body)
            body2["model"] = target_model
            return await forward(
                req=req,
                body=body2,
                req_id=req_id,
                session_id=session_id,
                bucket="chat",
                node_name=target_node,
                model_id=target_model,
                tried=(tried or []) + [f"{node_name}:{model_id}"],
                reason="backend_ctx_overflow_handover",
                started=started,
                extra_headers=extra_headers,
            )

        # Default: log and pass through backend error
        log.error(f"[backend_err] req_id={req_id} bucket={bucket} node={node_name} model={model_id} status={r.status_code} body={snippet}")
        return Response(content=raw, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


    # Success: pass through JSON
    raw = await r.aread()
    if _medical_demo_active and r.status_code < 400:
        try:
            raw = _apply_medical_output_disclaimer_json_bytes(raw)
            log.info(f"[{req_id}] → SCCI MEDICAL_DISCLAIMER_APPLIED sid={req_id} phase=output applied=1")
        except Exception as _medical_out_err:
            log.warning(f"[{req_id}] medical_disclaimer_output_err={type(_medical_out_err).__name__}:{_medical_out_err}")
    if _scci_longtext_tool_capture and r.status_code < 400:
        try:
            _scci_summary_text = _extract_assistant_text_from_chat_json_bytes(raw, max_chars=int(LONGTEXT_TOOL_SUMMARY_MAX_CHARS or 12000))
            if _scci_summary_text:
                _store_longtext_tool_summary(session, _scci_summary_text, _scci_longtext_tool_doc_hash, model_id)
                log.info(f"[{req_id}] → SCCI LONGTEXT_TOOL sid={req_id} phase=capture mode=nonstream summary_chars={len(_scci_summary_text)}")
        except Exception as _lt_nonstream_cap_err:
            log.warning(f"[{req_id}] SCCI longtext_tool_nonstream_capture_err={type(_lt_nonstream_cap_err).__name__}:{_lt_nonstream_cap_err}")
    resp = Response(content=raw, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    set_headers(resp, req_id, session_id, bucket, node_name, model_id, reason, tried, extra_headers)

    dt_ms = (time.time() - started) * 1000.0
    _update_latency_ema(node, bucket, dt_ms)
    if locals().get("session", None) is not None:
        _session_controller_complete(session)
    log.info(f"[{req_id}] → SCCI PROC_END sid={req_id} result=ok latency_ms={dt_ms:.0f} mode=nonstream")
    log.info(f"[{req_id}] → SCCI TIMELINE sid={req_id} state=1ms decision=1ms route=2ms exec={dt_ms:.0f}ms total={dt_ms:.0f}ms")
    log.info(f"[{req_id}] done total_ms={dt_ms:.0f} node_dt_ms={dt_ms:.0f}")
    return resp


def _update_latency_ema(node: Node, bucket: str, dt_ms: float) -> None:
    dt_ms = max(1.0, min(dt_ms, 600000.0))
    alpha = float(os.getenv("ROUTER_LAT_EMA_ALPHA", "0.2"))
    node.lat_ema_ms = (1.0 - alpha) * node.lat_ema_ms + alpha * dt_ms
    old = node.lat_ema_by_bucket.get(bucket, node.lat_ema_ms)
    node.lat_ema_by_bucket[bucket] = (1.0 - alpha) * old + alpha * dt_ms


# ===============================
# v29 - AI Telecom Ready Edition
# Hard Bucket Isolation Enabled
# ===============================


# ===============================
# v32 - clean architecture routing pass
# ===============================


# -----------------------------------------------------------------------------
# v12.7 DEMO IMPROVEMENT (LOG ONLY)
# Telecom-style cluster map printed at router startup.
# This does NOT modify routing logic, model selection, or cluster behavior.
# It only improves demo readability of the AI fabric topology.
# -----------------------------------------------------------------------------

def _scci_log_cluster_map():
    try:
        nodes = [
            "B-PHONE0(chat)",
            "B-PHONE1(memory)",
            "B-PHONE2(tools)",
            "B-CPU0(reasoning/coder)",
            "B-GPU0(vision/inference)",
            "B-GPU0-image-engine(diffusion)"
        ]

        log.info("SCCI CLUSTER_MAP")
        log.info(" | ".join(nodes))

    except Exception:
        pass


# Optional helper that can be called after cluster discovery initialization
def _scci_demo_startup_banner():
    try:
        log.info("LAB ROUTER v12.7")
        _scci_log_cluster_map()
        log.info("SCCI READY control-plane=active")
        log.info("SCCI MEDICAL_SAFETY demo_disclaimer=input+output classifier=strict")
    except Exception:
        pass




# ================================
# SCCI ROUTER V7 PATCH (Speculative Routing + Sleep Handling)
# ================================
import asyncio
import time

async def _send_request(node, payload, timeout):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(node.url("/v1/chat/completions"), json=payload)
            return {"ok": True, "response": r}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _detect_sleep(node_state):
    try:
        return "cooldown" in str(node_state).lower()
    except:
        return False

def _adaptive_timeout(node_state):
    return 2.0 if _detect_sleep(node_state) else 0.8

async def speculative_route(primary_node, fallback_nodes, payload):
    tasks = []
    timeout = _adaptive_timeout(primary_node)

    tasks.append(asyncio.create_task(_send_request(primary_node, payload, timeout)))

    for fb in fallback_nodes:
        await asyncio.sleep(0.15)
        tasks.append(asyncio.create_task(_send_request(fb, payload, 1.5)))

    for t in asyncio.as_completed(tasks):
        result = await t
        if result["ok"]:
            for other in tasks:
                if not other.done():
                    other.cancel()
            return result["response"]

    raise RuntimeError("All nodes failed")

# === END V7 PATCH ===


# ================================
# v9.6 CODING UPGRADE (NON-DESTRUCTIVE)
# ================================

import re

_ARCH_PAT_V96 = re.compile(
    r"\b(distributed|architecture|system design|scalable|cluster|multi-node|orchestrator|router|failover|fallback|pipeline)\b",
    re.IGNORECASE
)

def _detect_architecture_signal_v96(text: str) -> int:
    return 1 if text and _ARCH_PAT_V96.search(str(text)) else 0

def _pick_classifier_model_v96(user_text: str) -> str:
    t = str(user_text or "").lower()
    if any(k in t for k in ("design","architecture","distributed","system","router","pipeline","cluster")):
        return CODER_MEDIUM_MODEL_4K
    return CODER_SIMPLE_MODEL_4K

def _classify_coding_task_v96(user_text: str):
    heuristic = "simple_code"
    if len(user_text or "") > CODING_COMPLEX_PROMPT_CHAR_MIN:
        heuristic = "complex_code"

    arch = _detect_architecture_signal_v96(user_text)

    llm = "medium_code" if arch else heuristic

    final = llm
    if arch:
        final = "complex_code"
    elif heuristic == "complex_code":
        final = "complex_code"

    return final

