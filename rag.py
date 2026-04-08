import asyncio
import logging
import os
import pickle
import re
import string
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# OPTIONAL DEPS
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except ImportError:
    BM25_OK = False

try:
    import networkx as nx
    NX_OK = True
except ImportError:
    NX_OK = False

try:
    from langdetect import detect as langdetect_detect, LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False
    logger.warning("[WARN] langdetect not installed — Banglish detection will be limited.")


DEVICE = "cpu"

# CONFIG
API_BASE    = os.getenv("API_BASE",    "https://ewu-server.onrender.com/api")
API_KEY     = os.getenv("API_KEY",     "")
API_HEADERS = {"x-api-key": API_KEY}
GITHUB_BASE = os.getenv("GITHUB_BASE", "https://raw.githubusercontent.com/Atkiya/jsonfiles/main/")


LOCAL_GEN_MODEL  = os.getenv("LOCAL_GEN_MODEL",  "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
GEN_DEVICE       = os.getenv("GEN_DEVICE",       "cpu")
GEN_LOAD_IN_4BIT = os.getenv("GEN_LOAD_IN_4BIT", "false").lower() == "true"


_CPU_MAX_NEW_TOKENS = 256
_DEFAULT_MAX_TOKENS = _CPU_MAX_NEW_TOKENS if GEN_DEVICE == "cpu" else 700
GEN_MAX_NEW_TOKENS  = int(os.getenv("GEN_MAX_NEW_TOKENS", str(_DEFAULT_MAX_TOKENS)))

GEN_PROMPT_MAX_CHARS = int(os.getenv("GEN_PROMPT_MAX_CHARS", "3500"))

_CPU_TIMEOUT  = max(60, GEN_MAX_NEW_TOKENS * 2)   
GEN_TIMEOUT_S = int(os.getenv("GEN_TIMEOUT_S", str(_CPU_TIMEOUT if GEN_DEVICE == "cpu" else 90)))


GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
GROQ_MAX_TOKENS     = int(os.getenv("GROQ_MAX_TOKENS",  "700"))
GROQ_TIMEOUT        = int(os.getenv("GROQ_TIMEOUT",     "60"))


_default_prefer_groq = "true" if GEN_DEVICE == "cpu" else "false"
GEN_PREFER_GROQ = os.getenv("GEN_PREFER_GROQ", _default_prefer_groq).lower() == "true"

EMBED_MODEL           = os.getenv("EMBED_MODEL",           "intfloat/multilingual-e5-small")
PRIMARY_RERANK_MODEL  = os.getenv("PRIMARY_RERANK_MODEL",  "BAAI/bge-reranker-v2-m3")
FALLBACK_RERANK_MODEL = os.getenv("FALLBACK_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",       "560"))
TOP_K_RETRIEVE   = int(os.getenv("TOP_K_RETRIEVE",   "18"))
TOP_K_FINAL      = int(os.getenv("TOP_K_FINAL",      "6"))
RERANK_TOP_N     = int(os.getenv("RERANK_TOP_N",     "12"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-2.5"))
MMR_LAMBDA       = float(os.getenv("MMR_LAMBDA",     "0.75"))
CACHE_DIR        = os.getenv("CACHE_DIR",            "./cache")
CACHE_TTL_H      = int(os.getenv("CACHE_TTL_H",      "24"))
API_FAIL_LIMIT   = int(os.getenv("API_FAIL_LIMIT",   "3"))


CACHE_VERSION = "v8_tinyllama"

# API ENDPOINTS
API_LIST_ENDPOINTS = [
    ("admission-deadlines",  "Admission Deadlines",     {}),
    ("academic-calendar",    "Academic Calendar",       {}),
    ("grade-scale",          "Grade Scale",             {}),
    ("departments",          "Departments",             {}),
    ("programs",             "Programs (all)",          {}),
    ("programs",             "Programs (UG)",           {"level": "undergraduate"}),
    ("programs",             "Programs (Grad)",         {"level": "graduate"}),
    ("tuition-fees",         "Tuition Fees",            {}),
    ("tuition-fees",         "Tuition Fees (UG)",       {"level": "undergraduate"}),
    ("tuition-fees",         "Tuition Fees (Grad)",     {"level": "graduate"}),
    ("scholarships",         "Scholarships",            {}),
    ("clubs",                "Clubs",                   {}),
    ("events",               "Events",                  {}),
    ("notices",              "Notices",                 {"limit": 500}),
    ("helpdesk",             "Helpdesk",                {}),
    ("proctor-schedule",     "Proctor Schedule",        {}),
    ("governance",           "Governance (all)",        {}),
    ("governance",           "Board of Trustees",       {"body": "board_of_trustees"}),
    ("governance",           "Academic Council",        {"body": "academic_council"}),
    ("governance",           "Syndicate",               {"body": "syndicate"}),
    ("alumni",               "Alumni",                  {}),
    ("faculty",              "Faculty",                 {"limit": 200}),
    ("partnerships",         "Partnerships",            {}),
    ("policies",             "Policies",                {}),
    ("documents",            "Documents",               {}),
    ("newsletters",          "Newsletters",             {}),
    ("courses/programs",     "Course Programs (all)",   {}),
    ("courses/programs",     "Course Programs (UG)",    {"level": "undergraduate"}),
    ("courses/programs",     "Course Programs (Grad)",  {"level": "graduate"}),
    ("courses",              "Courses (all)",           {"limit": 500}),
    ("courses",              "Courses (UG)",            {"level": "undergraduate", "limit": 500}),
    ("courses",              "Courses (Grad)",          {"level": "graduate",      "limit": 500}),
    ("courses",              "Courses (Core)",          {"course_type": "Core",    "limit": 500}),
    ("courses",              "Courses (Elective)",      {"course_type": "Elective","limit": 500}),
]

API_DETAIL_CONFIGS = [
    {"list_suffix": "faculty",         "detail_prefix": "faculty",         "id_field": "id",            "label": "Faculty detail",        "params": {"limit": 200}},
    {"list_suffix": "documents",       "detail_prefix": "documents",       "id_field": "slug",          "label": "Document detail",       "params": {}},
    {"list_suffix": "courses/programs","detail_prefix": "courses/programs","id_field": "program_code",  "label": "Course-program detail", "params": {}},
    {"list_suffix": "programs",        "detail_prefix": "programs",        "id_field": "id",            "label": "Program detail",        "params": {}},
    {"list_suffix": "courses",         "detail_prefix": "courses",         "id_field": "course_code",   "label": "Course offering detail","params": {"limit": 500}},
]

GITHUB_FILES = [
    "dynamic_admission_process.json",
    "dynamic_admission_requirements.json",
    "dynamic_facilites.json",
    "ma_english.json",
    "mba_emba.json",
    "ms_cse.json",
    "ms_dsa.json",
    "mds.json",
    "mphil_pharmacy.json",
    "mss_eco.json",
    "scholarships_and_financial_aids.json",
    "st_ba.json",
    "st_ce.json",
    "st_cse.json",
    "st_ece.json",
    "st_economics.json",
    "st_eee.json",
    "st_english.json",
    "st_geb.json",
    "st_information_studies.json",
    "st_law.json",
    "st_math.json",
    "st_pharmacy.json",
    "st_social_relations.json",
    "st_sociology.json",
    "syndicate.json",
    "tesol.json",
    "ewu_board_of_trustees.json",
    "admission_deadlines.json",
    "ewu_faculty_complete.json",
    "dynamic_grading.json",
    "ewu_proctor_schedule.json",
    "ewu_newsletters_complete.json",
    "static_aboutEWU.json",
    "static_Admin.json",
    "static_AllAvailablePrograms.json",
    "static_alumni.json",
    "static_campus_life.json",
    "static_Career_Counseling_Center.json",
    "static_clubs.json",
    "static_depts.json",
    "static_facilities.json",
    "static_helpdesk.json",
    "static_payment_procedure.json",
    "static_Policy.json",
    "static_Programs.json",
    "static_Rules.json",
    "static_Sexual_harassment.json",
    "static_Tuition_fees.json",
]

DYNAMIC_HINTS = {
    "deadline", "deadlines", "admission", "fee", "fees", "tuition", "scholarship",
    "event", "events", "notice", "calendar", "contact", "email", "phone", "helpdesk",
    "department", "departments", "program", "programs",
}

DOMAIN_RULES = {
    "deadlines":   {"patterns": ["deadline","last date","admission deadline","apply date","shesh tarikh","শেষ তারিখ"],          "source_keywords": ["admission-deadlines","admission_deadlines","admission_process"]},
    "admission":   {"patterns": ["admission","apply","application","requirements","eligibility","document","ভর্তি","ডকুমেন্ট"],"source_keywords": ["admission","requirements","process","payment_procedure"]},
    "fees":        {"patterns": ["fee","fees","tuition","credit fee","waiver","scholarship","স্কলারশিপ","ফি","টিউশন"],           "source_keywords": ["tuition-fees","tution_fees","scholarships"]},
    "courses":     {"patterns": ["course","courses","program","programs","department","dept","credit","curriculum","department list","বিভাগ","কোর্স"],"source_keywords": ["courses","programs","depts","st_","static_programs"]},
    "departments": {"patterns": ["department","departments","dept","বিভাগ","বিভাগগুলো"],                                        "source_keywords": ["departments","dept","programs","bibhag"]},
    "contact":     {"patterns": ["contact","email","phone","helpdesk","registrar","office","যোগাযোগ","ইমেইল"],                  "source_keywords": ["helpdesk","admin","faculty"]},
    "events":      {"patterns": ["event","events","workshop","seminar","notice","news","month","এই মাসে"],                      "source_keywords": ["events","notices","newsletters"]},
    "location":    {"patterns": ["where","location","address","kothay","thikana","অবস্থিত","ঠিকানা"],                           "source_keywords": ["aboutEWU","campus","admin"]},
    "alumni":      {"patterns": ["alumni","former student","praktan","প্রাক্তন"],                                                "source_keywords": ["alumni"]},
    "facilities":  {"patterns": ["medical","hostel","dormitory","dorm","facility","facilities","center","medical center","হোস্টেল","মেডিকেল"],"source_keywords": ["facilites","campus_life","aboutEWU"]},
}

FIELD_PRIORITY = [
    "title","name","department","dept","program_name","program_title","program_code",
    "course_code","course_title","credit","credits","level","semester","duration",
    "deadline","date","start_date","end_date","email","phone","mobile","address",
    "location","amount","fee","tuition","description","eligibility","requirement","conditions",
]

EXACT_INTENTS = {"deadline","fee","credit","course_count","email","phone","location","contact"}

# APP STATE
class AppState:
    embedder              = None
    reranker              = None
    reranker_model_name: Optional[str] = None
    gen_model             = None
    gen_tokenizer         = None
    gen_model_name: str   = ""
    gen_mode: str         = "none"
    documents: List[Dict] = []
    faiss_index           = None
    doc_embeddings: Optional[np.ndarray] = None
    bm25                  = None
    bm25_tokens: List[List[str]] = []
    kg                    = None
    entity_index: Dict[str, List[int]] = {}
    ready: bool           = False
    error: str            = ""

state = AppState()
_api_fail_count: Dict[str, int] = {}
os.makedirs(CACHE_DIR, exist_ok=True)

# CACHE HELPERS
def _cp(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{CACHE_VERSION}_{name}")

def _cache_fresh(name: str) -> bool:
    p = _cp(name)
    if not os.path.exists(p):
        return False
    return (time.time() - os.path.getmtime(p)) / 3600 < CACHE_TTL_H

def _save_pickle(name: str, obj: Any) -> None:
    try:
        with open(_cp(name), "wb") as f:
            pickle.dump(obj, f, protocol=5)
        logger.info("[cache] saved %s", name)
    except Exception as e:
        logger.warning("[cache] save %s failed: %s", name, e)

def _load_pickle(name: str) -> Optional[Any]:
    try:
        with open(_cp(name), "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def _save_faiss(idx) -> None:
    if not FAISS_OK:
        return
    try:
        faiss.write_index(idx, _cp("faiss.index"))
        logger.info("[cache] saved faiss.index")
    except Exception as e:
        logger.warning("[cache] faiss save failed: %s", e)

def _load_faiss():
    if not FAISS_OK:
        return None
    p = _cp("faiss.index")
    if not os.path.exists(p):
        return None
    try:
        idx = faiss.read_index(p)
        logger.info("[cache] loaded faiss.index (%s vectors)", idx.ntotal)
        return idx
    except Exception:
        return None

# LANGUAGE + NORMALIZATION
_BANGLA_UNICODE_MIN = 0x0980
_BANGLA_UNICODE_MAX = 0x09FF

_BANGLISH_KEYWORDS = {
    "ami","tumi","apni","ki","koto","kothay","ache","hobe","hoy","jai","jabo","dao",
    "bolo","chai","lagbe","korte","kore","hoyeche","pabo","theke","porjonto","amar",
    "tomar","kivabe","kibhabe","vorti","bhorti","thikana","britti","bibhag","shikkharthi",
    "fee","result","admit","scholarship","credit","course",
}

BANGLISH_CANONICAL = {
    "vorti":"admission","bhorti":"admission","admit":"admission",
    "abedon":"application","abedoner":"application","prokriya":"process",
    "document":"documents","documents":"documents","dokument":"documents",
    "shesh":"last","shomoyshima":"deadline","shomoysima":"deadline","tarikh":"date",
    "kivabe":"how","kibhabe":"how","ki":"what","kothay":"where","thikana":"address",
    "bibhag":"department","course":"course","coursegulo":"courses","koyti":"count",
    "koto":"amount","fee":"fee","fees":"fee","tuition":"tuition","britti":"scholarship",
    "scholarship":"scholarship","helpdesk":"helpdesk","jogajog":"contact","credit":"credit",
    "shikkharthi":"student","international":"international","hostel":"hostel",
    "dormitory":"dormitory","alumni":"alumni","medical":"medical","registrar":"registrar",
}

NORMALIZE_RULES: List[Tuple[str, str]] = [
    (r"\bvc\b",                                  "vice chancellor"),
    (r"vice\s+chancellor",                        "vice chancellor"),
    (r"উপাচার্য|ভিসি",                            "vice chancellor"),
    (r"\bupacharj[oa]?\b|\bupacharja\b|\bvice chansellor\b|\bvice chancellor\b", "vice chancellor"),
    (r"ভর্তি|এডমিশন",                             "admission"),
    (r"\bvorti\b|\bbhorti\b",                    "admission"),
    (r"বৃত্তি|স্কলারশিপ",                         "scholarship"),
    (r"\bbritti\b|\bscholarship\b|\bwaiver\b",   "scholarship"),
    (r"ফি|খরচ|টিউশন",                            "tuition fee"),
    (r"\bfee\b|\bfees\b|\btuition\b|\bcost\b|\bkhoroch\b", "tuition fee"),
    (r"ঠিকানা|অবস্থান|কোথায়",                    "address location"),
    (r"\bthikana\b|\bkothay\b|\baddress\b|\blocation\b", "address location"),
    (r"ডেডলাইন|শেষ তারিখ|সময়সীমা",               "deadline"),
    (r"\bdeadline\b|\blast date\b|\bshomoyshima\b|\bshomoysima\b", "deadline"),
    (r"প্রয়োজনীয়তা|যোগ্যতা|শর্ত",               "requirements eligibility"),
    (r"\brequirement[s]?\b|\beligibility\b",     "requirements eligibility"),
    (r"বিভাগ|ডিপার্টমেন্ট",                       "department"),
    (r"\bbibhag\b|\bdepartment\b|\bdept\b",      "department"),
    (r"কোর্স|সাবজেক্ট",                           "course"),
    (r"\bsubject\b|\bcourse\b|\bcurriculum\b",   "course"),
    (r"নোটিশ|বিজ্ঞপ্তি",                          "notice"),
    (r"\bnotice\b|\bannouncement\b",             "notice"),
    (r"ইমেইল",                                    "email"),
    (r"যোগাযোগ",                                  "contact"),
]

def detect_language(text: str) -> str:
    if any(_BANGLA_UNICODE_MIN <= ord(ch) <= _BANGLA_UNICODE_MAX for ch in text):
        return "bangla"
    tokens = set(re.findall(r"\w+", text.lower()))
    if tokens & _BANGLISH_KEYWORDS:
        return "banglish"
    if LANGDETECT_OK:
        try:
            guess = langdetect_detect(text)
            if guess in ("bn", "hi"):
                return "banglish"
        except LangDetectException:
            pass
    return "english"

def _normalize_banglish_tokens(text: str) -> str:
    out = []
    for tok in re.findall(r"\w+|[^\w\s]", text.lower()):
        if re.fullmatch(r"\w+", tok):
            out.append(BANGLISH_CANONICAL.get(tok, tok))
        else:
            out.append(tok)
    return " ".join(out)

def normalize_query(text: str) -> str:
    q = text.strip().lower()
    q = _normalize_banglish_tokens(q)
    for pattern, repl in NORMALIZE_RULES:
        q = re.sub(pattern, repl, q, flags=re.IGNORECASE)
    q = re.sub(r"[^\w\s\u0980-\u09FF]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def tokenize_for_sparse(text: str) -> List[str]:
    norm = normalize_query(text)
    toks = [t for t in re.findall(r"[\w\u0980-\u09FF]+", norm) if len(t) > 1]
    return toks

# FETCHING
async def fetch_json(url: str, headers: dict = None, params: dict = None, timeout: int = 60):
    key = url.split("?")[0]
    if _api_fail_count.get(key, 0) >= API_FAIL_LIMIT:
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=headers or {}, params=params or {})
            if r.status_code == 200:
                _api_fail_count[key] = 0
                return r.json()
            if r.status_code in (404, 500, 502, 503):
                _api_fail_count[key] = _api_fail_count.get(key, 0) + 1
            logger.warning("[WARN] %s → HTTP %s", url, r.status_code)
    except Exception as e:
        _api_fail_count[key] = _api_fail_count.get(key, 0) + 1
        logger.warning("[WARN] %s → %s", url, e)
    return None

def _unwrap(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "results", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return []

async def _wake_api_server():
    logger.info("[API] Waking render.com server…")
    for attempt in range(3):
        result = await fetch_json(f"{API_BASE}/grade-scale", API_HEADERS, timeout=60)
        if result is not None:
            logger.info("[API] Server awake ✓")
            return True
        logger.info("[API] Wake attempt %s/3 failed, retrying in 10 s…", attempt + 1)
        await asyncio.sleep(10)
    logger.warning("[API] Server did not wake — continuing with GitHub only.")
    return False

def _source_tags(source: str) -> List[str]:
    source_low = source.lower()
    tags = []
    for domain, rule in DOMAIN_RULES.items():
        if any(k in source_low for k in rule["source_keywords"]):
            tags.append(domain)
    if any(x in source_low for x in (
        "admission","deadline","notice","fee","scholarship",
        "event","calendar","programs","departments",
    )):
        tags.append("dynamic")
    return sorted(set(tags))

async def load_api() -> list:
    if not await _wake_api_server():
        return []

    async def _fetch_list(suffix, label, params):
        url  = f"{API_BASE}/{suffix}"
        data = await fetch_json(url, API_HEADERS, params=params, timeout=60)
        items = _unwrap(data) if data is not None else []
        logger.info("[API] %-32s → %4s record(s)", label, len(items))
        return suffix, params, items

    list_results = await asyncio.gather(
        *[_fetch_list(s, l, p) for s, l, p in API_LIST_ENDPOINTS],
        return_exceptions=True,
    )

    docs: list = []
    list_cache: dict = {}
    for res in list_results:
        if isinstance(res, Exception):
            continue
        suffix, params, items = res
        for item in items:
            docs.append({"raw": item, "source": f"api:{suffix}", "source_tags": _source_tags(f"api:{suffix}")})
        if not params:
            list_cache.setdefault(suffix, items)

    detail_tasks = []
    for cfg in API_DETAIL_CONFIGS:
        base_items = list_cache.get(cfg["list_suffix"], [])
        for item in base_items:
            item_id = item.get(cfg["id_field"]) if isinstance(item, dict) else None
            if item_id:
                detail_tasks.append((
                    f"{API_BASE}/{cfg['detail_prefix']}/{item_id}",
                    f"api:{cfg['detail_prefix']}/{item_id}",
                ))

    sem = asyncio.Semaphore(5)

    async def _fetch_detail(url, source):
        async with sem:
            data  = await fetch_json(url, API_HEADERS, timeout=60)
            items = _unwrap(data) if data is not None else []
            return source, items

    if detail_tasks:
        detail_results = await asyncio.gather(
            *[_fetch_detail(u, s) for u, s in detail_tasks],
            return_exceptions=True,
        )
        for res in detail_results:
            if isinstance(res, Exception):
                continue
            source, items = res
            for item in items:
                docs.append({"raw": item, "source": source, "source_tags": _source_tags(source)})

    logger.info("[API] total raw docs: %s", len(docs))
    return docs

async def load_github() -> list:
    responses = await asyncio.gather(
        *[fetch_json(GITHUB_BASE + fname, timeout=60) for fname in GITHUB_FILES],
        return_exceptions=True,
    )
    docs = []
    for fname, data in zip(GITHUB_FILES, responses):
        if not data or isinstance(data, Exception):
            continue
        source = f"github:{fname}"
        for item in (data if isinstance(data, list) else [data]):
            docs.append({"raw": item, "source": source, "source_tags": _source_tags(source)})
    logger.info("[GitHub] total raw docs: %s", len(docs))
    return docs

# CHUNKING
def _flatten_json(obj, path="", sep=" > ") -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            np_ = f"{path}{sep}{k}" if path else k
            if isinstance(v, (dict, list)):
                lines.extend(_flatten_json(v, np_, sep))
            else:
                val = str(v).strip()
                if val and val.lower() not in ("null", "none", "", "[]", "{}"):
                    lines.append((np_, val))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_json(item, f"{path}[{i}]", sep))
            else:
                val = str(item).strip()
                if val:
                    lines.append((f"{path}[{i}]", val))
    return lines

def _prioritized_lines(flat_pairs: List[Tuple[str, str]]) -> List[str]:
    def score(k: str) -> Tuple[int, int]:
        key_low = k.lower()
        pri = 999
        for i, needle in enumerate(FIELD_PRIORITY):
            if needle in key_low:
                pri = i
                break
        return (pri, len(key_low))
    ordered = sorted(flat_pairs, key=lambda kv: score(kv[0]))
    return [f"{k}: {v}" for k, v in ordered]

def _make_record_text(raw: Any, source: str) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(raw, (dict, list)):
        text = str(raw).strip()
        return text, {"important_lines": [text] if text else []}

    flat            = _flatten_json(raw)
    important_lines = _prioritized_lines(flat)[:14]
    all_lines       = [f"{k}: {v}" for k, v in flat]

    header_bits = [source]
    if isinstance(raw, dict):
        for key in ("title","name","department","program_name","course_title","course_code"):
            if key in raw and str(raw[key]).strip():
                header_bits.append(f"{key}: {raw[key]}")
                break

    text_lines = ["source_label: " + " | ".join(header_bits)] + important_lines
    if len(text_lines) < 6:
        for line in all_lines:
            if line not in text_lines:
                text_lines.append(line)
            if len(" | ".join(text_lines)) >= CHUNK_SIZE:
                break

    text = " | ".join(text_lines)
    meta = {"important_lines": important_lines, "all_lines": all_lines, "flat_pairs": flat, "raw": raw}
    return text, meta

def chunk_documents(docs: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for doc in docs:
        raw    = doc.get("raw")
        source = doc["source"]
        text, meta = _make_record_text(raw, source)
        if not text.strip():
            continue
        out.append({"content": text, "source": source, "source_tags": doc.get("source_tags", []), "meta": meta})
    return out

# KNOWLEDGE GRAPH
_STOP = set(string.punctuation) | {
    "the","a","an","is","are","was","were","of","in","at","to","for",
    "and","or","not","this","that","it","its","with","as","by","on",
    "from","all","be","been","has","have","had","will","would","can",
    "could","do","does","did","he","she","they","we","you","i","me",
}

def build_knowledge_graph(docs: List[Dict]):
    if not NX_OK:
        return None, {}
    G = nx.DiGraph()
    entity_index: Dict[str, List[int]] = {}
    for ci, doc in enumerate(docs):
        pairs = doc.get("meta", {}).get("flat_pairs", [])
        for key, val in pairs:
            key_l, val_l = key.strip().lower(), str(val).strip().lower()
            if not key_l or not val_l:
                continue
            if not G.has_node(key_l): G.add_node(key_l, type="field")
            if not G.has_node(val_l): G.add_node(val_l, type="value")
            G.add_edge(key_l, val_l, chunk=ci)
            for tok in tokenize_for_sparse(val_l):
                if tok not in _STOP and len(tok) > 2:
                    entity_index.setdefault(tok, []).append(ci)
    logger.info("[KG] nodes=%s, edges=%s, tokens=%s", G.number_of_nodes(), G.number_of_edges(), len(entity_index))
    return G, entity_index

def kg_search(query: str, k: int = 5) -> List[int]:
    if not state.kg or not state.entity_index:
        return []
    tokens = [t for t in tokenize_for_sparse(query) if t not in _STOP]
    scores: Dict[int, int] = {}
    for tok in tokens:
        for idx in state.entity_index.get(tok, []):
            scores[idx] = scores.get(idx, 0) + 1
        if state.kg.has_node(tok):
            for nbr in state.kg.successors(tok):
                ed = state.kg[tok].get(nbr, {})
                ci = ed.get("chunk") if isinstance(ed, dict) else None
                if ci is not None:
                    scores[ci] = scores.get(ci, 0) + 1
    return sorted(scores, key=scores.get, reverse=True)[:k]

# INDEXING
def build_indexes_from_scratch() -> bool:
    if not state.documents:
        logger.warning("[WARN] No documents to index.")
        return False
    texts = [d["content"] for d in state.documents]

    if FAISS_OK and ST_OK and state.embedder:
        try:
            emb = state.embedder.encode(
                [f"passage: {t}" for t in texts],
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=64,
            )
            emb = np.array(emb, dtype="float32")
            if emb.ndim == 2 and emb.shape[0] > 0:
                idx = faiss.IndexFlatIP(emb.shape[1])
                idx.add(emb)
                state.faiss_index    = idx
                state.doc_embeddings = emb
                _save_faiss(idx)
                _save_pickle("doc_embeddings.pkl", emb)
                logger.info("[FAISS] %s vectors", idx.ntotal)
        except Exception as e:
            logger.error("[ERROR] FAISS: %s", e)

    if BM25_OK:
        try:
            tok = [tokenize_for_sparse(t) for t in texts if t.strip()]
            tok = [t for t in tok if t]
            if tok:
                state.bm25_tokens = tok
                state.bm25        = BM25Okapi(tok)
                _save_pickle("bm25.pkl", tok)
                _save_pickle("bm25_tokens.pkl", tok)
                logger.info("[BM25] %s docs", len(tok))
        except Exception as e:
            logger.error("[ERROR] BM25: %s", e)

    kg, ei = build_knowledge_graph(state.documents)
    state.kg           = kg
    state.entity_index = ei
    if kg is not None:
        _save_pickle("kg.pkl", kg)
        _save_pickle("entity_index.pkl", ei)

    _save_pickle("documents.pkl", state.documents)
    return True

def load_indexes_from_cache() -> bool:
    docs = _load_pickle("documents.pkl")
    if not docs:
        return False
    state.documents = docs

    idx = _load_faiss()
    if idx is not None:
        state.faiss_index = idx
    emb = _load_pickle("doc_embeddings.pkl")
    if emb is not None:
        state.doc_embeddings = emb

    bm25 = _load_pickle("bm25.pkl")
    if bm25 is not None:
        state.bm25 = bm25
    state.bm25_tokens = _load_pickle("bm25_tokens.pkl") or []

    kg           = _load_pickle("kg.pkl")
    entity_index = _load_pickle("entity_index.pkl")
    if kg is not None:
        state.kg           = kg
        state.entity_index = entity_index or {}

    return bool(state.documents) and (state.faiss_index is not None or state.bm25 is not None)


def detect_intent(query: str) -> str:
    q      = normalize_query(query)
    toks   = set(tokenize_for_sparse(q))
    q_text = " " + q + " "

    if any(x in q_text for x in (" email ", " ইমেইল ")):              return "email"
    if any(x in q_text for x in (" phone ", " mobile ", " number ")): return "phone"
    if "deadline" in toks or "date" in toks:                           return "deadline"
    if {"tuition", "fee"} & toks or "tuition fee" in q:               return "fee"
    if "credit" in toks and ("count" not in toks):                     return "credit"
    if ("course" in toks or "courses" in toks) and ({"count","number","koyti"} & toks): return "course_count"
    if {"where", "address", "location"} & toks:                        return "location"
    if {"contact", "registrar", "helpdesk", "office"} & toks:         return "contact"
    return "general"

def detect_domains(query: str) -> List[str]:
    q       = normalize_query(query)
    domains = []
    for domain, rule in DOMAIN_RULES.items():
        if any(p in q for p in rule["patterns"]):
            domains.append(domain)
    if not domains:
        domains = ["admission", "fees", "courses", "contact"]
    return domains

def candidate_weight_for_query(doc: Dict, query: str, domains: List[str]) -> float:
    score  = 0.0
    source = doc.get("source", "").lower()
    tags   = set(doc.get("source_tags", []))
    q      = normalize_query(query)
    for dom in domains:
        if dom in tags:
            score += 2.5
        for src_hint in DOMAIN_RULES.get(dom, {}).get("source_keywords", []):
            if src_hint in source:
                score += 1.5
    if any(x in q for x in DYNAMIC_HINTS) and "dynamic" in tags:
        score += 0.75
    return score

# RETRIEVAL
def _encode_query(query: str) -> np.ndarray:
    return np.array(
        state.embedder.encode([f"query: {query}"], normalize_embeddings=True),
        dtype="float32",
    )

def _dense(q_vec: np.ndarray, k: int = TOP_K_RETRIEVE) -> List[Dict]:
    if not state.faiss_index:
        return []
    try:
        k_a = min(k, state.faiss_index.ntotal)
        scores, ids = state.faiss_index.search(q_vec, k_a)
        return [{**state.documents[i], "dense_score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]
    except Exception as e:
        logger.error("[ERROR] dense: %s", e)
        return []

def _sparse(query: str, k: int = TOP_K_RETRIEVE) -> List[Dict]:
    if not state.bm25:
        return []
    try:
        tokens = tokenize_for_sparse(query)
        if not tokens:
            return []
        scores = np.array(state.bm25.get_scores(tokens), dtype="float32")
        idx    = np.argsort(scores)[::-1][:min(k, len(scores))]
        return [{**state.documents[i], "sparse_score": float(scores[i])} for i in idx if scores[i] > 0]
    except Exception as e:
        logger.error("[ERROR] sparse: %s", e)
        return []

def rrf_fuse(lists: List[List[Dict]], weights: List[float], rrf_k: int = 60) -> List[Dict]:
    merged, doc_map = {}, {}
    for lst, w in zip(lists, weights):
        for rank, d in enumerate(lst):
            key = d["source"] + "||" + d["content"]
            merged[key] = merged.get(key, 0.0) + w / (rrf_k + rank + 1)
            doc_map[key] = d
    return [
        {**doc_map[c], "rrf_score": round(s, 6)}
        for c, s in sorted(merged.items(), key=lambda x: x[1], reverse=True)
    ]

def expand_queries(query: str) -> List[str]:
    q_original = query.strip()
    q_norm     = normalize_query(q_original)
    variants   = [q_original, q_norm]

    expanded = q_norm
    acronyms = {
        "ewu":"east west university","cse":"computer science engineering",
        "eee":"electrical electronic engineering","ece":"electronic communication engineering",
        "bba":"bachelor of business administration","gpa":"grade point average",
        "cgpa":"cumulative grade point average","vc":"vice chancellor","dept":"department",
    }
    for abbr, full in acronyms.items():
        expanded = re.sub(r"\b" + re.escape(abbr) + r"\b", full, expanded, flags=re.IGNORECASE)
    variants.append(expanded)

    stop_q = {"what","who","when","where","how","why","is","are","does","do","the","a","an","tell","me","about"}
    kw = [w for w in tokenize_for_sparse(q_norm) if w not in stop_q and len(w) > 2]
    if kw:
        variants.append(" ".join(kw))

    out = []
    for v in variants:
        v = re.sub(r"\s+", " ", v).strip()
        if v and v not in out:
            out.append(v)
    return out[:4]

def rerank(query_for_rerank: str, candidates: List[Dict], top_n: int) -> List[Dict]:
    if not candidates:
        return []
    if not state.reranker:
        return candidates[:top_n]
    try:
        pairs  = [(query_for_rerank, d["content"]) for d in candidates]
        scores = state.reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        out    = [{**doc, "rerank_score": float(sc)} for sc, doc in scored[:top_n] if sc >= RERANK_MIN_SCORE]
        return out or [{**doc, "rerank_score": float(sc)} for sc, doc in scored[:top_n]]
    except Exception as e:
        logger.warning("[WARN] rerank unavailable: %s", e)
        return candidates[:top_n]

def mmr_select(q_vec: np.ndarray, candidates: List[Dict], k: int) -> List[Dict]:
    if state.doc_embeddings is None or not candidates:
        return candidates[:k]
    c2i  = {d["source"] + "||" + d["content"]: i for i, d in enumerate(state.documents)}
    keys = [d["source"] + "||" + d["content"] for d in candidates]
    idxs = [c2i[key] for key in keys if key in c2i]
    if not idxs:
        return candidates[:k]
    ce  = state.doc_embeddings[idxs]
    q   = q_vec[0]
    rel = ce @ q
    selected, sel_embs, remaining = [], [], list(range(len(idxs)))
    for _ in range(min(k, len(remaining))):
        if not sel_embs:
            best = max(remaining, key=lambda i: rel[i])
        else:
            S = np.array(sel_embs)
            best, best_score = remaining[0], -1e9
            for i in remaining:
                score = MMR_LAMBDA * rel[i] - (1 - MMR_LAMBDA) * float(np.max(S @ ce[i]))
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        sel_embs.append(ce[best])
        remaining.remove(best)
    return [candidates[i] for i in selected]

def _filter_by_domain(candidates: List[Dict], query: str, domains: List[str]) -> List[Dict]:
    rescored = []
    for doc in candidates:
        bonus    = candidate_weight_for_query(doc, query, domains)
        combined = doc.get("rrf_score", 0.0) + bonus
        rescored.append({**doc, "route_score": combined})
    rescored.sort(
        key=lambda d: (d.get("route_score", 0.0), d.get("rerank_score", -1e9), d.get("rrf_score", 0.0)),
        reverse=True,
    )
    return rescored

async def full_retrieval(query: str, k: int = TOP_K_FINAL) -> List[Dict]:
    variants = await asyncio.to_thread(expand_queries, query)
    domains  = detect_domains(query)
    all_dense, all_sparse = [], []

    for i, v in enumerate(variants):
        weight = 1.0 / (i + 1)
        if state.embedder:
            vec   = await asyncio.to_thread(_encode_query, v)
            dense = await asyncio.to_thread(_dense, vec, TOP_K_RETRIEVE)
            all_dense.append((dense, weight))
        sparse = await asyncio.to_thread(_sparse, v, TOP_K_RETRIEVE)
        all_sparse.append((sparse, weight))

    fused = rrf_fuse(
        [x for x, _ in all_dense] + [x for x, _ in all_sparse],
        [w for _, w in all_dense] + [w for _, w in all_sparse],
    )

    kg_idxs  = await asyncio.to_thread(kg_search, normalize_query(query), k * 3)
    existing = {d["source"] + "||" + d["content"] for d in fused}
    for i in kg_idxs:
        if 0 <= i < len(state.documents):
            d   = state.documents[i]
            key = d["source"] + "||" + d["content"]
            if key not in existing:
                fused.append({**d, "rrf_score": 0.0, "kg_injected": True})

    routed           = await asyncio.to_thread(_filter_by_domain, fused, query, domains)
    query_for_rerank = normalize_query(query)
    reranked         = await asyncio.to_thread(
        rerank,
        query_for_rerank,
        routed[: max(RERANK_TOP_N * 2, 14)],
        min(RERANK_TOP_N, max(k * 3, k)),
    )

    if state.embedder:
        q_vec = await asyncio.to_thread(_encode_query, normalize_query(query))
        return await asyncio.to_thread(mmr_select, q_vec, reranked, k)
    return reranked[:k]

# EXACT ANSWER EXTRACTION
def _best_matching_lines(doc: Dict, keywords: List[str], max_lines: int = 3) -> List[str]:
    all_lines = doc.get("meta", {}).get("all_lines", [])
    scored    = []
    for line in all_lines:
        l     = normalize_query(line)
        score = sum(1 for kw in keywords if kw in l)
        if score > 0:
            scored.append((score, line))
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    out = []
    for _, line in scored:
        if line not in out:
            out.append(line)
        if len(out) >= max_lines:
            break
    return out

def extract_structured_answer(query: str, lang: str, docs: List[Dict]) -> Optional[Dict[str, Any]]:
    if not docs:
        return None
    intent = detect_intent(query)
    if intent not in EXACT_INTENTS:
        return None

    keywords = {
        "deadline":     ["deadline","last date","date","admission"],
        "fee":          ["fee","tuition","credit fee","amount","credit","cse","computer science"],
        "credit":       ["credit","credits","total credits","duration"],
        "course_count": ["course","courses","count","department","cse"],
        "email":        ["email","mail","contact","helpdesk","registrar","cse"],
        "phone":        ["phone","mobile","telephone","contact","registrar"],
        "location":     ["address","location","campus","where"],
        "contact":      ["contact","email","phone","office","registrar","helpdesk"],
    }[intent]

    evidence = []
    for doc in docs:
        lines = _best_matching_lines(doc, keywords, max_lines=4)
        if lines:
            evidence.extend(lines)
        if len(evidence) >= 4:
            break

    if not evidence:
        return None

    evidence = evidence[:4]
    if lang == "bangla":
        lead = "প্রাসঙ্গিক তথ্য:"
    elif lang == "banglish":
        lead = "Relevant tothyo:"
    else:
        lead = "Relevant information:"

    answer = lead + "\n- " + "\n- ".join(evidence)
    return {"answer": answer, "intent": intent, "used_extraction": True, "evidence_lines": evidence}

# GENERATION
SYSTEM_PROMPT = (
    "You are EWU Assistant for East West University, Dhaka, Bangladesh. "
    "Answer based on the provided context. Use the most relevant facts from it. "
    "If the context contains partial information, use what is available and note any gaps. "
    "Only say you don't have information if the context is completely unrelated to the question. "
    "Never invent specific numbers, dates, names, or fees not present in the context. "
    "Give clear, natural, conversational answers — not raw data dumps. "
    "For lists of items, use bullet points. For single facts, use one sentence. "
    "You support English, Bangla (বাংলা), and Banglish responses."
)

def _lang_instruction(lang: str) -> str:
    if lang == "bangla":
        return "Respond entirely in Bengali (Bangla script, বাংলা)."
    if lang == "banglish":
        return "Respond entirely in Banglish — Bengali meaning written with English letters."
    return "Respond in clear English."

def _context_fallback(lang: str = "english") -> str:
    website = "https://www.ewubd.edu/"
    if lang == "bangla":
        return (
            "দুঃখিত, এই মুহূর্তে উত্তর তৈরি করা সম্ভব হচ্ছে না। "
            f"অনুগ্রহ করে আবার চেষ্টা করুন অথবা EWU ওয়েবসাইট দেখুন: {website}"
        )
    if lang == "banglish":
        return (
            "Sorry, ekhon answer generate kora jacche na. "
            f"Please abar try korun othoba EWU website visit korun: {website}"
        )
    return (
        "I'm sorry, I couldn't generate a response at this moment. "
        f"Please try again or visit the EWU website: {website}"
    )

def _weak_evidence(results: List[Dict]) -> bool:
    if not results:
        return True
    top = results[0]
    if top.get("rerank_score") is not None and top.get("rerank_score") < -4.0:
        return True
    return False

# Local TinyLlama generation 
def _generate_local_sync(query: str, context: str, lang: str) -> str:
    """
    Run TinyLlama-1.1B-Chat-v1.0 inference synchronously.

    TinyLlama uses the Zephyr/HF chat template:
        <|system|>\\n{system}</s>\\n<|user|>\\n{user}</s>\\n<|assistant|>\\n
    apply_chat_template() handles this automatically from the tokenizer config.

    do_sample=True + low temperature keeps output factual while avoiding the
    "generation flags not valid" warning caused by greedy + saved config mismatch.
    """
    import torch

    model     = state.gen_model
    tokenizer = state.gen_tokenizer
    if model is None or tokenizer is None:
        logger.error("[GEN] Model or tokenizer is None — skipping local generation.")
        return ""

    trimmed_ctx = context[:GEN_PROMPT_MAX_CHARS]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + _lang_instruction(lang)},
        {"role": "user",   "content": f"Context:\n{trimmed_ctx}\n\nQuestion: {query}"},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
       
        inputs    = tokenizer(text, return_tensors="pt", truncation=True, max_length=1600)
        device    = next(model.parameters()).device
        inputs    = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        logger.info("[GEN] TinyLlama prompt tokens: %d, max_new: %d", prompt_len, GEN_MAX_NEW_TOKENS)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,        
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][prompt_len:]
        if len(new_ids) == 0:
            logger.warning("[GEN] TinyLlama generated 0 new tokens.")
            return ""

        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        logger.info("[GEN] TinyLlama: %d tokens → %d chars", len(new_ids), len(answer))
        return answer

    except Exception as e:
        logger.error("[GEN] TinyLlama local generation failed: %s", e, exc_info=True)
        return ""

#fallback Groq generation (used when local model is unavailable or slow)
async def _generate_groq(query: str, context: str, lang: str) -> str:
    if not GROQ_API_KEY:
        logger.warning("[GEN] Groq requested but GROQ_API_KEY is not set.")
        return _context_fallback(lang)

    trimmed_ctx  = context[:GEN_PROMPT_MAX_CHARS]
    user_content = (
        f"{_lang_instruction(lang)}\n\n"
        f"Context:\n{trimmed_ctx}\n\n"
        f"Question: {query}"
    )

    try:
        async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model":       GROQ_FALLBACK_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    "max_tokens":  GROQ_MAX_TOKENS,
                    "temperature": 0.1,
                },
            )

           
            if r.status_code != 200:
                logger.error(
                    "[GEN] Groq HTTP %s — body: %s",
                    r.status_code,
                    r.text[:600],
                )
                return _context_fallback(lang)

            answer = r.json()["choices"][0]["message"]["content"].strip()
            return answer or _context_fallback(lang)

    except httpx.TimeoutException:
        logger.error("[GEN] Groq request timed out after %ds.", GROQ_TIMEOUT)
        return _context_fallback(lang)
    except Exception as e:
        logger.error("[GEN] Groq inference error: %s", e)
        return _context_fallback(lang)


async def generate(query: str, context: str, lang: str) -> str:
    """
    GEN_PREFER_GROQ=true  (default on CPU): Groq → TinyLlama → fallback text
    GEN_PREFER_GROQ=false (GPU preferred):  TinyLlama → Groq → fallback text
    """

    async def _try_local() -> str:
        if state.gen_model is None:
            return ""
        try:
            logger.info("[GEN] TinyLlama local: max_new=%d timeout=%ds",
                        GEN_MAX_NEW_TOKENS, GEN_TIMEOUT_S)
            answer = await asyncio.wait_for(
                asyncio.to_thread(_generate_local_sync, query, context, lang),
                timeout=GEN_TIMEOUT_S,
            )
            if answer:
                logger.info("[GEN] ✓ TinyLlama answered (%d chars).", len(answer))
            else:
                logger.warning("[GEN] TinyLlama returned empty string.")
            return answer
        except asyncio.TimeoutError:
            logger.error("[GEN] TinyLlama timed out after %ds.", GEN_TIMEOUT_S)
            return ""
        except Exception as e:
            logger.error("[GEN] TinyLlama error: %s", e)
            return ""

    async def _try_groq() -> str:
        if not GROQ_API_KEY:
            logger.warning("[GEN] Groq requested but GROQ_API_KEY is not set.")
            return ""
        logger.info("[GEN] Using Groq (%s).", GROQ_FALLBACK_MODEL)
        return await _generate_groq(query, context, lang)

    if GEN_PREFER_GROQ:
        answer = await _try_groq()
        if answer:
            return answer
        answer = await _try_local()
        if answer:
            return answer
    else:
        answer = await _try_local()
        if answer:
            return answer
        answer = await _try_groq()
        if answer:
            return answer

    logger.error("[GEN] All backends failed — returning static fallback.")
    return _context_fallback(lang)

# MODEL LOADING
def _load_gen_model():
    """
    Load TinyLlama/TinyLlama-1.1B-Chat-v1.0.
    Returns (model, tokenizer) or (None, None) on failure.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info("[GEN] Loading local SLM: %s  device=%s  4bit=%s",
                    LOCAL_GEN_MODEL, GEN_DEVICE, GEN_LOAD_IN_4BIT)

        torch_dtype = torch.float32 if GEN_DEVICE == "cpu" else torch.float16

        model_kwargs: Dict[str, Any] = {
            "torch_dtype":       torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if GEN_DEVICE not in ("cpu", "mps"):
            model_kwargs["device_map"] = "auto"

        if GEN_LOAD_IN_4BIT and GEN_DEVICE not in ("cpu", "mps"):
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("[GEN] 4-bit quantization enabled.")
            except ImportError:
                logger.warning("[GEN] bitsandbytes not installed — skipping 4-bit.")

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_GEN_MODEL, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(LOCAL_GEN_MODEL, **model_kwargs)

        if "device_map" not in model_kwargs:
            model = model.to(GEN_DEVICE)

        model.eval()
        logger.info("[GEN] ✓ TinyLlama ready on %s.", GEN_DEVICE)
        return model, tokenizer

    except Exception as e:
        logger.error("[GEN] Could not load local gen model (%s): %s", LOCAL_GEN_MODEL, e)
        return None, None


def _load_retrieval_models():
    emb, reranker, reranker_name = None, None, None
    if ST_OK:
        try:
            logger.info("Loading embedder (%s) on %s…", EMBED_MODEL, DEVICE)
            emb = SentenceTransformer(EMBED_MODEL, device=DEVICE)
            logger.info("Embedder ready.")
        except Exception as e:
            logger.error("[ERROR] Embedder: %s", e)

        for model_name in (PRIMARY_RERANK_MODEL, FALLBACK_RERANK_MODEL):
            if not model_name:
                continue
            try:
                logger.info("Loading reranker (%s)…", model_name)
                reranker      = CrossEncoder(model_name, device=DEVICE, max_length=512)
                reranker_name = model_name
                logger.info("Reranker ready.")
                break
            except Exception as e:
                logger.warning("[WARN] Reranker %s unavailable: %s", model_name, e)
    return emb, reranker, reranker_name

# BOOT
async def _boot():
    try:
        logger.info("=== BOOT: EWU RAG (local SLM=%s, device=%s) ===", LOCAL_GEN_MODEL, GEN_DEVICE)

        emb, reranker, reranker_name = await asyncio.to_thread(_load_retrieval_models)
        state.embedder            = emb
        state.reranker            = reranker
        state.reranker_model_name = reranker_name

        gen_model, gen_tokenizer = await asyncio.to_thread(_load_gen_model)
        if gen_model is not None:
            state.gen_model      = gen_model
            state.gen_tokenizer  = gen_tokenizer
            state.gen_model_name = LOCAL_GEN_MODEL
            state.gen_mode       = "groq+local" if (GEN_PREFER_GROQ and GROQ_API_KEY) else "local"
            logger.info("[GEN] TinyLlama loaded. prefer_groq=%s  device=%s  "
                        "max_new_tokens=%d  timeout=%ds",
                        GEN_PREFER_GROQ, GEN_DEVICE, GEN_MAX_NEW_TOKENS, GEN_TIMEOUT_S)
        elif GROQ_API_KEY:
            state.gen_mode       = "groq"
            state.gen_model_name = GROQ_FALLBACK_MODEL
            logger.warning("[GEN] Local model unavailable — Groq-only mode (%s).", GROQ_FALLBACK_MODEL)
        else:
            state.gen_mode = "none"
            logger.warning("[GEN] No local model AND no GROQ_API_KEY — answers will be fallback text.")

        cache_ok = (
            _cache_fresh("documents.pkl")
            and _cache_fresh("faiss.index")
            and _cache_fresh("bm25.pkl")
        )
        if cache_ok and await asyncio.to_thread(load_indexes_from_cache):
            logger.info("[cache] loaded %s chunks from disk.", len(state.documents))
            state.ready = True
            return

        logger.info("Fetching knowledge base (API + GitHub)…")
        api_docs, gh_docs = await asyncio.gather(load_api(), load_github())
        raw_docs = api_docs + gh_docs
        logger.info("Raw docs combined: %s", len(raw_docs))

        if not raw_docs:
            logger.warning("[WARN] No documents fetched.")
            state.ready = True
            return

        logger.info("Chunking documents…")
        state.documents = await asyncio.to_thread(chunk_documents, raw_docs)
        logger.info("Total chunks: %s", len(state.documents))

        logger.info("Building indexes…")
        await asyncio.to_thread(build_indexes_from_scratch)
        state.ready = True
        logger.info("✓ EWU RAG stack fully ready.")

    except Exception as e:
        state.error = str(e)
        state.ready = False
        logger.exception("[ERROR] Boot failed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_boot())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

# APP + ENDPOINTS
app = FastAPI(
    title="EWU RAG Server (TinyLlama-1.1B-Chat local + Groq fallback)",
    lifespan=lifespan,
)

class Query(BaseModel):
    query: str
    top_k: int = TOP_K_FINAL


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "service":     "EWU RAG Server",
            "status":      "ready" if state.ready else ("error" if state.error else "loading"),
            "docs_loaded": len(state.documents),
            "gen_mode":    state.gen_mode,
            "gen_model":   state.gen_model_name,
            "endpoints": {
                "POST /rag":   "Submit a question, get an answer",
                "GET /health": "Detailed health / model status",
                "GET /":       "This overview",
            },
            "hint": "POST to /rag with JSON body: {\"query\": \"your question\"}",
        },
    )


@app.post("/rag")
async def rag_endpoint(q: Query):
    if not state.ready:
        raise HTTPException(503, detail=state.error or "Still initializing — retry shortly.")
    raw_query = q.query.strip()
    if not raw_query:
        raise HTTPException(400, detail="Query must not be empty.")

    lang       = detect_language(raw_query)
    normalized = normalize_query(raw_query)
    intent     = detect_intent(raw_query)
    domains    = detect_domains(raw_query)
    results    = await full_retrieval(raw_query, k=max(1, min(q.top_k, 8)))

    if not results:
        return {
            "answer":            _context_fallback(lang),
            "detected_language": lang,
            "normalized_query":  normalized,
            "intent":            intent,
            "domains":           domains,
            "gen_mode":          state.gen_mode,
            "sources":           [],
        }

    if _weak_evidence(results):
        if lang == "bangla":
            answer = "প্রাসঙ্গিক তথ্যভিত্তিক নিশ্চিত উত্তর পাওয়া যায়নি।"
        elif lang == "banglish":
            answer = "Proshno-r jonno nishchit context-vittik uttor pawa jayni."
        else:
            answer = "I do not have confirmed information for that from the available context."
    else:
        context = "\n\n---\n\n".join(r["content"] for r in results)
        answer  = await generate(raw_query, context, lang)

    return {
        "answer":            answer,
        "detected_language": lang,
        "normalized_query":  normalized,
        "intent":            intent,
        "domains":           domains,
        "used_extraction":   False,
        "gen_mode":          state.gen_mode,
        "gen_model":         state.gen_model_name,
        "reranker_model":    state.reranker_model_name,
        "sources": [
            {
                "source":       r.get("source"),
                "source_tags":  r.get("source_tags", []),
                "rrf_score":    round(r.get("rrf_score", 0), 6),
                "rerank_score": round(r.get("rerank_score", 0), 4) if r.get("rerank_score") is not None else None,
                "route_score":  round(r.get("route_score", 0), 4) if r.get("route_score") is not None else None,
                "kg_injected":  r.get("kg_injected", False),
            }
            for r in results
        ],
    }


@app.get("/health")
async def health():
    return JSONResponse(200, {
        "status":                  "ready" if state.ready else ("error" if state.error else "loading"),
        "docs":                    len(state.documents),
        "retrieval_device":        DEVICE,
        "gen_mode":                state.gen_mode,
        "gen_model":               state.gen_model_name,
        "gen_device":              GEN_DEVICE,
        "gen_prefer_groq":         GEN_PREFER_GROQ,
        "gen_max_new_tokens":      GEN_MAX_NEW_TOKENS,
        "gen_timeout_s":           GEN_TIMEOUT_S,
        "local_gen_model":         LOCAL_GEN_MODEL,
        "embed_model":             EMBED_MODEL,
        "reranker_model":          state.reranker_model_name,
        "primary_reranker_model":  PRIMARY_RERANK_MODEL,
        "fallback_reranker_model": FALLBACK_RERANK_MODEL,
        "groq_fallback_model":     GROQ_FALLBACK_MODEL,
        "groq_key_set":            bool(GROQ_API_KEY),
        "faiss":                   state.faiss_index is not None,
        "bm25":                    state.bm25 is not None,
        "reranker":                state.reranker is not None,
        "local_pipeline_loaded":   state.gen_model is not None,
        "error":                   state.error or None,
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
