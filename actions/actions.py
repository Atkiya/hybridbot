import asyncio
from typing import Any, Text, Dict, List
import logging
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import requests
import re
import json

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_embedder = SentenceTransformer("intfloat/multilingual-e5-small")


def embed_query(text: str) -> list:
    return _embedder.encode(f"query: {text}", normalize_embeddings=True).tolist()


API_BASE_URL = "https://ewu-server.onrender.com/api"
API_KEY      = "i6EDytaX4E2jI6GvZQc0b1RSZHTI5_wVRa2rfL7rLpk"
HEADERS      = {"x-api-key": API_KEY}

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Atkiya/jsonfiles/main"
GITHUB_DATA_SOURCES = {
    "admission_process": "dynamic_admission_process.json",
    "facilities":        "dynamic_facilites.json",
    "static_programs":   "static_Programs.json",
    # Graduate program course files
    "ma_english":        "ma_english.json",
    "mba_emba":          "mba_emba.json",
    "ms_cse":            "ms_cse.json",
    "ms_dsa":            "ms_dsa.json",
    "mds":               "mds.json",
    "mphil_pharmacy":    "mphil_pharmacy.json",
    "mss_eco":           "mss_eco.json",
    # Undergraduate program course files
    "bba":               "bba.json",
    "st_ba":             "st_ba.json",
    "tesol":             "tesol.json",    
    "st_cse":            "st_cse.json",
    "st_eee":            "st_eee.json",
    "st_ce":             "st_ce.json",
    "st_english":        "st_english.json",
    "st_pharmacy":       "st_pharmacy.json",
    "st_law":            "st_law.json",
    "st_economics":      "st_economics.json",
    "st_sociology":      "st_sociology.json",
    "helpdesk_contacts": "static_helpdesk.json"
}


# LANGUAGE HELPERS


_BANGLA_UNICODE_MIN = 0x0980
_BANGLA_UNICODE_MAX = 0x09FF

_BANGLISH_KEYWORDS = {
    "ami", "tumi", "apni", "ki", "koto", "kothay", "ache", "hobe",
    "hoy", "jai", "jabo", "dao", "nao", "bolo", "dekho", "jano",
    "chai", "lagbe", "korte", "kore", "hoyeche", "pabo", "dite", "nite",
    "theke", "boro", "chhoto", "valo", "bhalo", "kharap", "ektu", "onek",
    "shob", "keu", "kono", "amar", "tomar", "tar", "eta", "ota",
    "ekhane", "okhane", "kobe", "keno", "kemne", "kivabe",
}

def detect_language(text: str) -> str:
    """Returns 'bangla', 'banglish', or 'english'."""
    if any(_BANGLA_UNICODE_MIN <= ord(ch) <= _BANGLA_UNICODE_MAX for ch in text):
        return "bangla"
    try:
        from langdetect import detect as ld, LangDetectException
        if ld(text) in ("bn", "hi"):
            return "banglish"
    except Exception:
        pass
    if set(text.lower().split()) & _BANGLISH_KEYWORDS:
        return "banglish"
    return "english"


_LANG_PREFIX = {
    "bangla":   " EWU তথ্য:\n\n",
    "banglish": " EWU Info:\n\n",
    "english":  "",
}
_LANG_SUFFIX = {
    "bangla":   "\n\n(বিস্তারিত জানতে: admissions@ewubd.edu |  09666775577)",
    "banglish": "\n\n(Beshi info er jonno: admissions@ewubd.edu |  09666775577)",
    "english":  "",
}
_NO_INFO_MSG = {
    "bangla":   "দুঃখিত, এই বিষয়ে কোনো তথ্য পাওয়া যায়নি।",
    "banglish": "Khed, ei bishoy e kono info pai ni.",
    "english":  "Sorry, I couldn't find information on that.",
}
_CONTACT_MSG = {
    "bangla": (
        "দুঃখিত, সুনির্দিষ্ট তথ্য পাওয়া যায়নি। আপনি:\n"
        "• ওয়েবসাইট দেখুন: https://www.ewubd.edu\n"
        "• ইমেইল করুন: admissions@ewubd.edu\n"
        "• ফোন করুন: 09666775577, Ext. 234"
    ),
    "banglish": (
        "Sorry, specific info pailam na. Apni:\n"
        "• Website: https://www.ewubd.edu\n"
        "• Email: admissions@ewubd.edu\n"
        "• Phone: 09666775577, Ext. 234"
    ),
    "english": (
        "I'm sorry, I couldn't find specific information about that. You can:\n"
        "• Visit our website: https://www.ewubd.edu\n"
        "• Contact Admissions: admissions@ewubd.edu\n"
        "• Call: 09666775577, Ext. 234 | +8801755587224 (Hotline)"
    ),
}

_LOCALIZE_REPLACEMENTS = {
    "bangla": {
        " EWU Undergraduate Admission — Application Steps": " EWU স্নাতক ভর্তি — আবেদন করার ধাপসমূহ",
        " EWU Graduate Admission — Application Steps": " EWU স্নাতকোত্তর ভর্তি — আবেদন করার ধাপসমূহ",
        " EWU Online Admission — Step-by-Step Guide": " EWU অনলাইন ভর্তি — ধাপে ধাপে গাইড",
        " EWU Admission Application Fee": " EWU ভর্তি আবেদন ফি",
        " EWU Admission — Required Documents and Uploads": " EWU ভর্তি — প্রয়োজনীয় কাগজপত্র ও আপলোড",
        " EWU Admission — Login and EWU Login ID": " EWU ভর্তি — লগইন ও EWU লগইন আইডি",
        " EWU Admission Test — Admit Card and Instructions": " EWU ভর্তি পরীক্ষা — অ্যাডমিট কার্ড ও নির্দেশনা",
        " EWU Admission — How to Fill and Submit the Online Form": " EWU ভর্তি — অনলাইন ফর্ম কীভাবে পূরণ ও জমা দিতে হবে",
        " East West University Location": " ইস্ট ওয়েস্ট ইউনিভার্সিটির অবস্থান",
        " East West University Vision": " ইস্ট ওয়েস্ট ইউনিভার্সিটির ভিশন",
        " East West University Mission": " ইস্ট ওয়েস্ট ইউনিভার্সিটির মিশন",
        " East West University History": " ইস্ট ওয়েস্ট ইউনিভার্সিটির ইতিহাস",
        " East West University Departments": " ইস্ট ওয়েস্ট ইউনিভার্সিটির বিভাগসমূহ",
        " East West University Programs": " ইস্ট ওয়েস্ট ইউনিভার্সিটির প্রোগ্রামসমূহ",
        " Available Course Programs": " উপলভ্য কোর্স প্রোগ্রামসমূহ",
        " Course Information": " কোর্স তথ্য",
        " EWU Grading System": " EWU গ্রেডিং সিস্টেম",
        " EWU Tuition Fees Overview": " EWU টিউশন ফি সংক্ষিপ্তসার",
        " EWU Scholarships": " EWU স্কলারশিপসমূহ",
        " EWU Helpdesk Contacts": " EWU হেল্পডেস্ক যোগাযোগসমূহ",
        " EWU Admission Deadlines": " EWU ভর্তি আবেদনের শেষ তারিখসমূহ",
        " EWU Admission Requirements": " EWU ভর্তি যোগ্যতা ও শর্তাবলি",
        " EWU General Conduct Rules": " EWU সাধারণ আচরণবিধি",
        " EWU Clubs and Societies": " EWU ক্লাব ও সংগঠনসমূহ",
        " EWU Alumni Network": " EWU অ্যালামনাই নেটওয়ার্ক",
        " EWU Recent Events": " EWU ইভেন্টসমূহ",
        " EWU Latest Notices": " EWU সর্বশেষ নোটিশসমূহ",
        " EWU Governance": " EWU পরিচালনা কাঠামো",
        " EWU International Partnerships": " EWU আন্তর্জাতিক পার্টনারশিপসমূহ",
        " EWU Newsletters": " EWU নিউজলেটারসমূহ",
        " EWU Proctor Schedule": " EWU প্রক্টর সময়সূচি",
        " EWU Exam Schedule": " EWU পরীক্ষার সময়সূচি",
        " EWU Academic Calendar": " EWU একাডেমিক ক্যালেন্ডার",
        "Admission Office": "ভর্তি অফিস",
        "Address": "ঠিকানা",
        "Phone": "ফোন",
        "Mobile": "মোবাইল",
        "Email": "ইমেইল",
        "Website": "ওয়েবসাইট",
        "Web": "ওয়েব",
        "Code": "কোড",
        "Title": "শিরোনাম",
        "Credits": "ক্রেডিট",
        "Prerequisites": "পূর্বশর্ত",
        "Description": "বিবরণ",
        "Program": "প্রোগ্রাম",
        "Programs": "প্রোগ্রামসমূহ",
        "Type": "ধরণ",
        "Section": "সেকশন",
        "Total returned": "মোট পাওয়া গেছে",
        "Faculty of Science and Engineering": "বিজ্ঞান ও প্রকৌশল অনুষদ",
        "Faculty of Business and Economics": "ব্যবসা ও অর্থনীতি অনুষদ",
        "Faculty of Arts and Social Sciences": "কলা ও সমাজবিজ্ঞান অনুষদ",
        "Other Academic Units": "অন্যান্য একাডেমিক ইউনিট",
        "Undergraduate Programs (Per Credit)": "স্নাতক প্রোগ্রামসমূহ (প্রতি ক্রেডিট)",
        "Graduate Programs (Per Credit)": "স্নাতকোত্তর প্রোগ্রামসমূহ (প্রতি ক্রেডিট)",
        "Undergraduate": "স্নাতক",
        "Graduate": "স্নাতকোত্তর",
        "Postgraduate": "স্নাতকোত্তর",
        "Grade Scale": "গ্রেড স্কেল",
        "Special Grades": "বিশেষ গ্রেড",
        "Eligibility": "যোগ্যতা",
        "Amount/Waiver": "পরিমাণ/ওয়েভার",
        "Min CGPA": "ন্যূনতম সিজিপিএ",
        "Freedom Fighter's Scholarship": "মুক্তিযোদ্ধা কোটার স্কলারশিপ",
        "Graduate Scholarships": "স্নাতকোত্তর স্কলারশিপ",
        "Based on Bachelor's CGPA": "স্নাতক সিজিপিএর ভিত্তিতে",
        "Academic": "একাডেমিক",
        "Administrative": "প্রশাসনিক",
        "Dept": "বিভাগ",
        "Purpose": "উদ্দেশ্য",
        "Deadline": "শেষ তারিখ",
        "Test": "পরীক্ষা",
        "Required Documents": "প্রয়োজনীয় কাগজপত্র",
        "Bachelor's Degree": "স্নাতক ডিগ্রি",
        "SSC & HSC GPA": "এসএসসি ও এইচএসসি জিপিএ",
        "SSC & HSC": "এসএসসি ও এইচএসসি",
        "Diploma": "ডিপ্লোমা",
        "Expected Behavior": "প্রত্যাশিত আচরণ",
        "Academic Misconduct (Zero Tolerance)": "একাডেমিক অসদাচরণ (শূন্য সহনশীলতা)",
        "Social Misconduct": "সামাজিক অসদাচরণ",
        "Notable Alumni": "উল্লেখযোগ্য অ্যালামনাই",
        "Achievement": "অর্জন",
        "Position": "পদবি",
        "Country": "দেশ",
        "Partnership": "পার্টনারশিপ",
        "Chairperson": "চেয়ারপারসন",
        "Step-by-Step Guide": "ধাপে ধাপে গাইড",
        "Application Steps": "আবেদন করার ধাপসমূহ",
        "Application Fee": "আবেদন ফি",
        "Step": "ধাপ",
        "Note": "নোট",
        "Tip": "পরামর্শ",
        "Need help?": "সহায়তা দরকার?",
        "Recent": "সাম্প্রতিক",
        "Format": "ফরম্যাট",
        "Max size": "সর্বোচ্চ সাইজ",
        "Printed Admit Card": "প্রিন্ট করা অ্যাডমিট কার্ড",
        "Original": "মূল",
        "Example": "উদাহরণ",
        "Save": "সেভ",
        "Preview": "প্রিভিউ",
        "Submit Form": "ফর্ম সাবমিট",
        "No course data found for": "এই বিভাগের কোনো কোর্স তথ্য পাওয়া যায়নি:",
        "No": "না",
        "Yes": "হ্যাঁ",
    },
    "banglish": {
        " EWU Undergraduate Admission — Application Steps": " EWU Undergraduate Admission — Apply korar steps",
        " EWU Graduate Admission — Application Steps": " EWU Graduate Admission — Apply korar steps",
        " EWU Online Admission — Step-by-Step Guide": " EWU Online Admission — Step by step guide",
        " EWU Admission Application Fee": " EWU Admission Application Fee",
        " EWU Admission — Required Documents and Uploads": " EWU Admission — Required documents o uploads",
        " EWU Admission — Login and EWU Login ID": " EWU Admission — Login o EWU Login ID",
        " EWU Admission Test — Admit Card and Instructions": " EWU Admission Test — Admit card o instructions",
        " EWU Admission — How to Fill and Submit the Online Form": " EWU Admission — Online form kivabe fill-up o submit korben",
        " East West University Location": " East West University er location",
        " East West University Vision": " East West University er vision",
        " East West University Mission": " East West University er mission",
        " East West University History": " East West University er history",
        " East West University Departments": " East West University er department gulo",
        " East West University Programs": " East West University er program gulo",
        " Available Course Programs": " Available course program gulo",
        " Course Information": " Course info",
        " EWU Grading System": " EWU grading system",
        " EWU Tuition Fees Overview": " EWU tuition fees overview",
        " EWU Scholarships": " EWU scholarship gulo",
        " EWU Helpdesk Contacts": " EWU helpdesk contacts",
        " EWU Admission Deadlines": " EWU admission deadline gulo",
        " EWU Admission Requirements": " EWU admission requirements",
        " EWU General Conduct Rules": " EWU general conduct rules",
        " EWU Clubs and Societies": " EWU clubs o societies",
        " EWU Alumni Network": " EWU alumni network",
        " EWU Recent Events": " EWU recent events",
        " EWU Latest Notices": " EWU latest notices",
        " EWU Governance": " EWU governance",
        " EWU International Partnerships": " EWU international partnerships",
        " EWU Newsletters": " EWU newsletters",
        " EWU Proctor Schedule": " EWU proctor schedule",
        " EWU Exam Schedule": " EWU exam schedule",
        " EWU Academic Calendar": " EWU academic calendar",
        "Admission Office": "Admission office",
        "Address": "Thikana",
        "Phone": "Phone",
        "Mobile": "Mobile",
        "Email": "Email",
        "Website": "Website",
        "Web": "Web",
        "Code": "Code",
        "Title": "Title",
        "Credits": "Credits",
        "Prerequisites": "Prerequisites",
        "Description": "Description",
        "Program": "Program",
        "Programs": "Program gulo",
        "Type": "Type",
        "Section": "Section",
        "Total returned": "Mot pawa geche",
        "Faculty of Science and Engineering": "Faculty of Science and Engineering",
        "Faculty of Business and Economics": "Faculty of Business and Economics",
        "Faculty of Arts and Social Sciences": "Faculty of Arts and Social Sciences",
        "Other Academic Units": "Onnanno academic units",
        "Undergraduate Programs (Per Credit)": "Undergraduate program gulo (per credit)",
        "Graduate Programs (Per Credit)": "Graduate program gulo (per credit)",
        "Undergraduate": "Undergraduate",
        "Graduate": "Graduate",
        "Postgraduate": "Postgraduate",
        "Grade Scale": "Grade scale",
        "Special Grades": "Special grades",
        "Eligibility": "Eligibility",
        "Amount/Waiver": "Amount/Waiver",
        "Min CGPA": "Min CGPA",
        "Freedom Fighter's Scholarship": "Freedom Fighter scholarship",
        "Graduate Scholarships": "Graduate scholarships",
        "Based on Bachelor's CGPA": "Bachelor's CGPA er upor vitti kore",
        "Academic": "Academic",
        "Administrative": "Administrative",
        "Dept": "Dept",
        "Purpose": "Purpose",
        "Deadline": "Deadline",
        "Test": "Test",
        "Required Documents": "Required documents",
        "Bachelor's Degree": "Bachelor's degree",
        "SSC & HSC GPA": "SSC & HSC GPA",
        "SSC & HSC": "SSC & HSC",
        "Diploma": "Diploma",
        "Expected Behavior": "Expected behavior",
        "Academic Misconduct (Zero Tolerance)": "Academic misconduct (zero tolerance)",
        "Social Misconduct": "Social misconduct",
        "Notable Alumni": "Notable alumni",
        "Achievement": "Achievement",
        "Position": "Position",
        "Country": "Country",
        "Partnership": "Partnership",
        "Chairperson": "Chairperson",
        "Step-by-Step Guide": "Step by step guide",
        "Application Steps": "Application steps",
        "Application Fee": "Application fee",
        "Step": "Step",
        "Note": "Note",
        "Tip": "Tip",
        "Need help?": "Help lagbe?",
        "Recent": "Recent",
        "Format": "Format",
        "Max size": "Max size",
        "Printed Admit Card": "Printed admit card",
        "Original": "Original",
        "Example": "Example",
        "Save": "Save",
        "Preview": "Preview",
        "Submit Form": "Submit form",
        "No course data found for": "Ei department er kono course data pawa jay ni:",
        "No": "No",
        "Yes": "Yes",
    },
}


def _localize_template(text: str, lang: str) -> str:
    """Best-effort template localization for Bangla/Banglish outputs."""
    if lang not in ("bangla", "banglish") or not text:
        return text
    localized = text
    replacements = _LOCALIZE_REPLACEMENTS.get(lang, {})
    for src in sorted(replacements, key=len, reverse=True):
        localized = localized.replace(src, replacements[src])
    localized = re.sub(r"\bStep\s+(\d+)\s*:", ("ধাপ \1:" if lang == "bangla" else "Step \1:"), localized)
    localized = re.sub(r"\bPhone\s*:", ("ফোন:" if lang == "bangla" else "Phone:"), localized)
    localized = re.sub(r"\bEmail\s*:", ("ইমেইল:" if lang == "bangla" else "Email:"), localized)
    localized = re.sub(r"\bAddress\s*:", ("ঠিকানা:" if lang == "bangla" else "Thikana:"), localized)
    localized = re.sub(r"\bWebsite\s*:", ("ওয়েবসাইট:" if lang == "bangla" else "Website:"), localized)
    localized = re.sub(r"\bDeadline\s*:", ("শেষ তারিখ:" if lang == "bangla" else "Deadline:"), localized)
    localized = re.sub(r"\bProgram\s*:", ("প্রোগ্রাম:" if lang == "bangla" else "Program:"), localized)
    localized = re.sub(r"\bDescription\s*:", ("বিবরণ:" if lang == "bangla" else "Description:"), localized)
    return localized


def _lang_wrap(text: str, lang: str) -> str:
    """Wrap a response with language-appropriate prefix/suffix."""
    text = _localize_template(text, lang)
    return _LANG_PREFIX.get(lang, "") + text + _LANG_SUFFIX.get(lang, "")


def get_user_language(tracker: Tracker) -> str:
    lang = tracker.get_slot("user_language")
    if lang in ("bn", "banglish", "en", "bangla", "english"):
        return {"bn": "bangla", "en": "english"}.get(lang, lang)
    for e in tracker.latest_message.get("entities", []):
        if e.get("entity") == "user_language":
            val = e.get("value", "en")
            return {"bn": "bangla", "en": "english"}.get(val, val)
    return detect_language(tracker.latest_message.get("text", ""))


def fetch_api_data(endpoint: str, params: Dict[str, Any] = None) -> Any:
    """Live API fetch — used only as a fallback when preloaded DATA is missing."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        print(f"\n[API FETCH] GET {url}  params={params}")
        response = requests.get(url, headers=HEADERS, params=params, timeout=60)
        print(f"[API FETCH] Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"[API FETCH] Response preview: {json.dumps(data, ensure_ascii=False)[:500]}")
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data
        else:
            logger.error(f"Failed to fetch {endpoint}: Status {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Exception fetching {endpoint}: {e}")
        return []


def fetch_detailed_data(list_endpoint: str, id_field: str, detail_endpoint_prefix: str) -> Dict[str, Any]:
    print(f"\n[DETAILED FETCH] Starting: {list_endpoint}")
    items = fetch_api_data(list_endpoint)
    if not isinstance(items, list):
        items = [items] if items else []
    detailed_results = {}
    for item in items:
        item_id = item.get(id_field)
        if item_id:
            try:
                detail = fetch_api_data(f"{detail_endpoint_prefix}/{item_id}")
                detailed_results[str(item_id)] = detail
            except Exception as e:
                logger.error(f"Failed detail for {item_id}: {e}")
    print(f"[DETAILED FETCH] Done: {len(detailed_results)} records for {list_endpoint}")
    return detailed_results


def load_from_github(filename: str) -> Any:
    """Fetch JSON data from the configured GitHub raw source."""
    try:
        url = f"{GITHUB_RAW_BASE}/{filename}"
        print(f"\n[GITHUB FETCH] GET {url}")
        response = requests.get(url, timeout=30)
        print(f"[GITHUB FETCH] Status: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        logger.error(f"Failed to fetch GitHub file {filename}: Status {response.status_code}")
        return {}
    except Exception as e:
        logger.error(f"Exception fetching GitHub file {filename}: {e}")
        return {}


def utter_smart(dispatcher: CollectingDispatcher, tracker: Tracker, text: str) -> None:
    """Dispatch language-aware responses using the existing language wrappers."""
    lang = get_user_language(tracker)
    dispatcher.utter_message(text=_lang_wrap(text, lang))



# STARTUP DATA LOAD


try:
    print("\n" + "=" * 60)
    print("LOADING EWU DATA FROM API")
    print("=" * 60)
    DATA: Dict[str, Any] = {}
    _simple_endpoints = {
        "admission_deadlines": "admission-deadlines",
        "academic_calendar":   "academic-calendar",
        "grade_scale":         "grade-scale",
        "tuition_fees":        "tuition-fees",
        "scholarships":        "scholarships",
        "clubs":               "clubs",
        "notices":             "notices",
        "partnerships":        "partnerships",
        "governance":          "governance",
        "alumni":              "alumni",
        "helpdesk_db":         "helpdesk",
        "policies":            "policies",
        "proctor_schedule":    "proctor-schedule",
        "newsletters":         "newsletters",
        "programs":            "programs",
        "faculty":             "faculty",
        "departments":         "departments",
        "documents":           "documents",
        "events":              "events",
        "course_programs_ug":  "courses/programs",
        "courses_all":         "courses",
    }
    for key, endpoint in _simple_endpoints.items():
        DATA[key] = fetch_api_data(endpoint)
    DATA["admission_process"] = load_from_github(GITHUB_DATA_SOURCES["admission_process"])
    DATA["static_programs"] = load_from_github(GITHUB_DATA_SOURCES["static_programs"])
    DATA["facilities"] = load_from_github(GITHUB_DATA_SOURCES["facilities"])
    DATA["helpdesk_contacts"] = load_from_github(GITHUB_DATA_SOURCES["helpdesk_contacts"])
    DATA["programs_list"] = DATA["programs"]
    DATA["faculty_list"]  = DATA["faculty"]
    DATA["programs_map"]  = fetch_detailed_data("programs",  "id",   "programs")
    DATA["faculty_map"]   = fetch_detailed_data("faculty",   "id",   "faculty")
    DATA["documents_map"] = fetch_detailed_data("documents", "slug", "documents")
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60 + "\n")
except Exception as e:
    print(f"CRITICAL ERROR loading data: {e}")
    DATA = {}



# HELPERS



 
DEPARTMENT_MAPPING = {
    "cse": "st_cse", "computer science": "st_cse",
    "eee": "st_eee", "electrical": "st_eee",
    "bba": "bba",    "business": "bba",
    "ce": "st_ce",   "civil": "st_ce",
    "english": "st_english", "pharmacy": "st_pharmacy",
    "law": "st_law",
    "economics": "st_economics", "eco": "st_economics",
    "sociology": "st_sociology", "soc": "st_sociology",
}
 
GRADUATE_MAPPING = {
    "ms cse":       "ms_cse",
    "ms dsa":       "ms_dsa", "data science": "ms_dsa",
    "mba":          "mba_emba", "emba": "mba_emba",
}
 
GRADUATE_PROGRAM_CODES = ["MS-CSE","MS-DSA","MBA","EMBA","MA-ENG","MA-TESOL","LL.M","M.PHARM","MSS-ECO","MDS"]



def _get_document(slug: str) -> Dict[str, Any]:
    """
    Return a document by slug from the preloaded documents_map.
    Falls back to a live API call only if the map is empty or the slug is missing.
    """
    doc = DATA.get("documents_map", {}).get(slug)
    if doc:
        return doc
    logger.warning(f"[_get_document] slug '{slug}' not in preloaded documents_map — fetching live")
    return fetch_api_data(f"documents/{slug}") or {}


def get_course_by_code(course_code: str) -> Dict[str, Any]:
    """
    Search for a course by code across all GitHub-backed program data files.
    Undergraduate: DEPARTMENT_MAPPING values → DATA[key]
    Graduate:      GRADUATE_MAPPING values   → DATA[key]
    No API calls made.
    """
    course_code = course_code.upper().strip()
 
    #  Undergraduate
    for data_key in DEPARTMENT_MAPPING.values():
        program_data = DATA.get(data_key)
        if not program_data or not isinstance(program_data, dict):
            continue
 
        dept_info    = program_data.get("department_info", {})
        program_name = dept_info.get("program_name") or dept_info.get("department_name") or ""
 
        # Main course list
        for course in program_data.get("courses", []):
            if course.get("code", "").upper() == course_code:
                return {
                    "success":       True,
                    "course_code":   course_code,
                    "course_title":  course.get("name") or course.get("title", ""),
                    "credits":       course.get("credits", 0),
                    "prerequisites": course.get("prerequisites", "None"),
                    "description":   course.get("description", ""),
                    "program":       program_name,
                    "major":         "",
                    "category":      course.get("category", ""),
                }
 
        # Major-specific lists — check both field name variants
        for major_name, major_data in program_data.get("majors", {}).items():
            core_courses = (
                major_data.get("required_courses", [])    
                + major_data.get("compulsory_courses", [])
            )
            for course in core_courses:
                if course.get("code", "").upper() == course_code:
                    return {
                        "success":       True,
                        "course_code":   course_code,
                        "course_title":  course.get("name") or course.get("title", ""),
                        "credits":       course.get("credits", 0),
                        "prerequisites": course.get("prerequisites", "None"),
                        "description":   course.get("description", ""),
                        "program":       program_name,
                        "major":         major_name,
                        "category":      f"Core - {major_name}",
                    }
            for course in major_data.get("elective_courses", []):
                if course.get("code", "").upper() == course_code:
                    return {
                        "success":       True,
                        "course_code":   course_code,
                        "course_title":  course.get("name") or course.get("title", ""),
                        "credits":       course.get("credits", 0),
                        "prerequisites": course.get("prerequisites", "None"),
                        "description":   course.get("description", ""),
                        "program":       program_name,
                        "major":         major_name,
                        "category":      f"Elective - {major_name}",
                    }
 
    #  Graduate
    for data_key in GRADUATE_MAPPING.values():
        program_data = DATA.get(data_key)
        if not program_data or not isinstance(program_data, dict):
            continue
        program_name = program_data.get("program", "")
        for course in program_data.get("courses", []):
            if course.get("code", "").upper() == course_code:
                return {
                    "success":       True,
                    "course_code":   course_code,
                    "course_title":  course.get("name") or course.get("title", ""),
                    "credits":       course.get("credits", 0),
                    "prerequisites": course.get("prerequisites", "None"),
                    "description":   course.get("description", ""),
                    "program":       program_name,
                    "major":         "",
                    "category":      course.get("category", ""),
                }
 
    return {"success": False, "message": f"Course {course_code} not found"}
 





# ACTION CLASSES


class ActionGetLocation(Action):
    def name(self): return "action_get_location"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        logger.info(f"[action_get_location] lang={lang}")

        doc = _get_document("about-ewu")
        if not doc or not isinstance(doc, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        content = doc.get("content", {})
        address = content.get("address", {})
        contact = content.get("contact", {})
        full_address = (
            f"{address.get('street_address','')}, {address.get('area','')}, "
            f"{address.get('city','')} - {address.get('post_code','')}, "
            f"{address.get('country','')}"
        )
        core = (
            f"Address: {full_address}\n"
            f" Phone: {contact.get('phone','')}\n"
            f" Email: {contact.get('email','')}\n"
            f" Website: {contact.get('website','')}"
        )
        dispatcher.utter_message(text=_lang_wrap(" East West University Location:\n\n" + core, lang))
        return []


class ActionGetSocialMedia(Action):
    def name(self): return "action_get_social_media"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        logger.info(f"[action_get_social_media] lang={lang}")
        core = (
            " Website: www.ewubd.edu\n📧 Admissions: admissions@ewubd.edu\n"
            " Phone: 09666775577, Ext. 234\n\n"
            " Facebook:  https://www.facebook.com/myewu/\n"
            "   YouTube:   https://www.youtube.com/EastWestUniversity96\n"
            "   LinkedIn:  https://www.linkedin.com/school/my-east-west-university/\n"
            "   Twitter:   https://x.com/myewubd\n"
            "   Instagram: https://www.instagram.com/myewu96/"
        )
        dispatcher.utter_message(text=_lang_wrap(core, lang))
        return []


class ActionGetEWUHistory(Action):
    def name(self): return "action_get_ewu_history"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        logger.info(f"[action_get_ewu_history] lang={lang}")

        doc = _get_document("about-ewu")
        if not doc or not isinstance(doc, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        history = doc.get("content", {}).get("history", {})
        if not history:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        core  = " East West University History:\n\n"
        core += f" Idea: {history.get('idea','')}\n"
        core += f" Lead Founder: {history.get('lead_founder','')}\n"
        core += f" Founded By: {history.get('founding_organization','')}\n"
        core += f" Legal Basis: {history.get('legal_basis','')}\n\n"
        core += f" Launch Year: {history.get('launch_year','')}\n"
        core += f" First Classes: {history.get('first_classes_start_date','')}\n"
        core += f" Initial Faculty: {history.get('initial_faculty','')}\n"
        core += f" Initial Students: {history.get('initial_students','')}\n\n"
        core += f" Current Faculty: {history.get('current_faculty','')}\n"
        core += f" Current Students: {history.get('current_students','')}\n"
        core += f" Initial Campus: {history.get('initial_campus_location','')}"
        dispatcher.utter_message(text=_lang_wrap(core, lang))
        return []


class ActionGetEWUVision(Action):
    def name(self): return "action_get_ewu_vision"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        doc = _get_document("about-ewu")
        if not doc or not isinstance(doc, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        vision_list = doc.get("content", {}).get("vision", [])
        if not vision_list:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        core = " East West University Vision:\n\n" + "".join(f"• {p}\n\n" for p in vision_list)
        dispatcher.utter_message(text=_lang_wrap(core.strip(), lang))
        return []


class ActionGetEWUMission(Action):
    def name(self): return "action_get_ewu_mission"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        doc = _get_document("about-ewu")
        if not doc or not isinstance(doc, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        mission_list = doc.get("content", {}).get("mission", [])
        if not mission_list:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        core = " East West University Mission:\n\n" + "".join(
            f"{i}. {p}\n\n" for i, p in enumerate(mission_list, 1)
        )
        dispatcher.utter_message(text=_lang_wrap(core.strip(), lang))
        return []


class ActionListDepartments(Action):
    def name(self): return "action_list_departments"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        departments = DATA.get("departments") or fetch_api_data("departments")
        if not departments:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        seen: Dict[str, Dict] = {}
        for dept in departments:
            key = f"{dept.get('faculty','')}||{dept.get('name','')}"
            if key not in seen or dept.get("code"):
                seen[key] = dept

        grouped: Dict[str, List] = {}
        for dept in seen.values():
            grouped.setdefault(dept.get("faculty", "Other"), []).append(dept)

        message = " East West University Departments:\n\n"
        faculty_order = [
            "Faculty of Science and Engineering",
            "Faculty of Business and Economics",
            "Faculty of Arts and Social Sciences",
            "Other Academic Units",
        ]

        def _fmt(d):
            code = d.get("code") or ""
            name = d.get("name", "Unknown")
            return f"   • {name} ({code.upper()})\n" if code else f"   • {name}\n"

        for fac in faculty_order:
            if fac in grouped:
                message += f" {fac}:\n" + "".join(_fmt(d) for d in grouped[fac]) + "\n"
        for fac, depts in grouped.items():
            if fac not in faculty_order:
                message += f" {fac}:\n" + "".join(_fmt(d) for d in depts) + "\n"

        dispatcher.utter_message(text=_lang_wrap(message.strip(), lang))
        return []

        
GITHUB_COURSE_PROGRAM_MAP = {
    # Undergraduate
    "bba":                    "bba",
    "business administration": "bba",
    "ba english":             "st_ba",
    "ba in english":          "st_ba",
    # Graduate
    "ms cse":                 "ms_cse",
    "msc cse":                "ms_cse",
    "ms computer":            "ms_cse",
    "master of science in computer": "ms_cse",
    "ms dsa":                 "ms_dsa",
    "ms data":                "ms_dsa",
    "data science":           "ms_dsa",
    "mba":                    "mba_emba",
    "emba":                   "mba_emba",
    "executive mba":          "mba_emba",
    "ma english":             "ma_english",
    "tesol":                  "tesol",          
    "teaching english":       "tesol",          
    "ma tesol":               "tesol",
    "ma in english":          "ma_english",
    "tesol":                  "ma_english",
    "mds":                    "mds",
    "development studies":    "mds",
    "mphil":                  "mphil_pharmacy",
    "m pharm":                "mphil_pharmacy",
    "mss eco":                "mss_eco",
    "mss economics":          "mss_eco",
    "master of social science": "mss_eco",
}

 
_github_course_files = [
    # Graduate
    "ma_english", "tesol", "mba_emba", "ms_cse", "ms_dsa",
    "mds", "mphil_pharmacy", "mss_eco",
    # Undergraduate
    "bba", "st_ba",
    "st_cse", "st_eee", "st_ce",
    "st_english", "st_pharmacy", "st_law",
    "st_economics", "st_sociology",
]
for _key in _github_course_files:
    _loaded = load_from_github(GITHUB_DATA_SOURCES.get(_key, f"{_key}.json"))
    DATA[f"gh_courses_{_key}"] = _loaded   
    DATA[_key]                 = _loaded 




# GITHUB COURSE HELPERS


def _normalize_course_list(raw: List) -> List[Dict]:
    """Normalize varied field names into a consistent course dict."""
    normalized = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        code    = (c.get("course_code") or c.get("code") or c.get("course") or "").strip()
        title   = (c.get("course_title") or c.get("title") or c.get("name") or "").strip()
        credits = c.get("credits") or c.get("credit") or c.get("credit_hours") or "?"

       
        if not title and code and not code[:3].isdigit():
            parts = code.split(" ", 2)
            if len(parts) >= 2 and parts[0][-1].isdigit():
                code  = parts[0]
                title = " ".join(parts[1:])

        if code or title:
            normalized.append({
                "course_code":  code  or "N/A",
                "course_title": title or "N/A",
                "credits":      credits,
            })
    return normalized


def _extract_courses_from_github(data: Any, program_label: str = "") -> List[Dict]:
    if not data:
        return []

 
    if isinstance(data, list):
        return _normalize_course_list(data)

    if not isinstance(data, dict):
        return []


    if "courses" in data and isinstance(data["courses"], list):
        return _normalize_course_list(data["courses"])

   
    collected = []
    for val in data.values():
        if isinstance(val, list):
            collected.extend(_normalize_course_list(val))
        elif isinstance(val, dict):
            inner = val.get("courses") or val.get("course_list") or []
            if isinstance(inner, list):
                collected.extend(_normalize_course_list(inner))
            else:
                for v2 in val.values():
                    if isinstance(v2, list):
                        collected.extend(_normalize_course_list(v2))
    seen = set()
    unique = []
    for c in collected:
        key = c["course_code"]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique




class ActionGetCourses(Action):
    def name(self): return "action_get_courses"
 
    def run(self, dispatcher, tracker, domain):
        lang        = get_user_language(tracker)
        user_message = tracker.latest_message.get("text", "")
        user_lower  = user_message.lower()
 
        #  1. Specific course-code lookup 
        matches = re.findall(r'\b([A-Z]{2,4}\s*\d{3,4})\b', user_message.upper())
        if matches:
            result = get_course_by_code(matches[0].replace(" ", ""))
            if result["success"]:
                msg  = " Course Information:\n\n"
                msg += f"Code:          {result['course_code']}\n"
                msg += f"Title:         {result['course_title']}\n"
                msg += f"Credits:       {result['credits']}\n"
                msg += f"Prerequisites: {result['prerequisites']}\n"
                if result.get("category"): msg += f"Category:      {result['category']}\n"
                if result.get("major"):    msg += f"Major:         {result['major']}\n"
                if result["description"]:  msg += f"\nDescription: {result['description'][:300]}..."
                msg += f"\n\nProgram: {result.get('program', 'N/A')}"
            else:
                msg = f" {result.get('message', 'Course not found')}. Please check the course code."
            dispatcher.utter_message(text=_lang_wrap(msg, lang))
            return []
 
        gh_key = None
        for phrase in sorted(GITHUB_COURSE_PROGRAM_MAP, key=len, reverse=True):
            if phrase in user_lower:
                gh_key = GITHUB_COURSE_PROGRAM_MAP[phrase]
                break
 
        if gh_key:
            raw_gh  = DATA.get(f"gh_courses_{gh_key}") or load_from_github(
                GITHUB_DATA_SOURCES.get(gh_key, f"{gh_key}.json")
            )
            courses = _extract_courses_from_github(raw_gh, gh_key)
            label   = gh_key.upper().replace("_", " ")
 
            if not courses:
                dispatcher.utter_message(
                    text=_lang_wrap(
                        f"Course details for {label} are not yet available. "
                        "Contact admissions@ewubd.edu or visit https://www.ewubd.edu.",
                        lang,
                    )
                )
            else:
                msg  = f" Courses — {label}:\n\nTotal: {len(courses)}\n\n"
                msg += "".join(
                    f"  {c['course_code']}: {c['course_title']} ({c['credits']} cr)\n"
                    for c in courses
                )
                dispatcher.utter_message(text=_lang_wrap(msg, lang))
            return []
 
        
        department = None
        for dept_alias in DEPARTMENT_MAPPING:
            if dept_alias in user_lower:
                department = dept_alias
                break
 
        if not department:
            slot_val = tracker.get_slot("department")
            if slot_val and slot_val.lower() in DEPARTMENT_MAPPING:
                department = slot_val.lower()
 
       
        if not department:
            msg       = " All Available Courses:\n\n"
            total_shown = 0
            seen_keys = set()
            for dept_alias, dept_key in DEPARTMENT_MAPPING.items():
                if dept_key in seen_keys:
                    continue
                seen_keys.add(dept_key)
                program_data = DATA.get(dept_key, {})
                if not isinstance(program_data, dict):
                    continue
                courses = program_data.get("courses", [])
                if not courses:
                    continue
                dept_info    = program_data.get("department_info", {})
                program_name = (
                    dept_info.get("program_name")
                    or dept_info.get("department_name")
                    or dept_alias.upper()
                )
                msg += f"🔹 {program_name}:\n"
                for course in courses:
                    msg += f"   {course.get('code','N/A')}: {course.get('name','N/A')} ({course.get('credits','?')} cr)\n"
               
                total_shown += 1
 
            if total_shown == 0:
                msg = (
                    " No course data is currently available.\n\n"
                    "Departments: CSE, EEE, BBA, Civil, English, Law, Pharmacy, Economics, Sociology\n\n"
                    "Example: 'Show me CSE courses' or 'Tell me about BUS101'"
                )
            dispatcher.utter_message(text=_lang_wrap(msg, lang))
            return []
 
        #  5. Show courses for the detected UG department 
        dept_key     = DEPARTMENT_MAPPING[department]
        program_data = DATA.get(dept_key, {})
 
        if not isinstance(program_data, dict) or not program_data:
            dispatcher.utter_message(
                text=_lang_wrap(f"No course data found for {department.upper()}.", lang)
            )
            return [SlotSet("department", department)]
 
        courses = program_data.get("courses", [])
 
     
        major_courses = 0
        for major_data in program_data.get("majors", {}).values():
            major_courses += len(major_data.get("required_courses", []))
            major_courses += len(major_data.get("compulsory_courses", []))
            major_courses += len(major_data.get("elective_courses", []))
 
        dept_info    = program_data.get("department_info", {})
        program_name = (
            dept_info.get("program_name")
            or dept_info.get("department_name")
            or department.upper()
        )
 
        msg  = f" {program_name}\n\n"
        msg += f"Total Courses: {len(courses) + major_courses}\n\n"
        msg += "Sample Courses:\n"
        for i, course in enumerate(courses, start=1):
            msg += f"{i}. {course.get('code','N/A')}: {course.get('name','N/A')} ({course.get('credits','?')} cr)\n"
       
        msg += "\n\n💡 Ask about a specific course: 'What is BUS101?'"
 
        dispatcher.utter_message(text=_lang_wrap(msg, lang))
        return [SlotSet("department", department)]
        
        

class ActionGetCourseDetails(Action):
    def name(self): return "action_get_course_details"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        course_code  = tracker.get_slot("course_code")
        user_message = tracker.latest_message.get("text", "")

        if not course_code:
            matches = re.findall(r'\b([A-Z]{2,4}\s*\d{3,4})\b', user_message.upper())
            if matches:
                course_code = matches[0].replace(" ", "")

        if not course_code:
            dispatcher.utter_message(
                text=_lang_wrap("Please specify a course code, e.g. CSE303 or ENG101.", lang)
            )
            return []

        result = get_course_by_code(course_code.upper())
        if result["success"]:
            msg = (
                f" {result['course_code']}: {result['course_title']}\n\n"
                f"Credits:       {result['credits']}\n"
                f"Prerequisites: {result['prerequisites']}\n"
            )
            if result.get("course_type"):
                msg += f"Type:          {result['course_type']}\n"
            if result.get("section"):
                msg += f"Section:       {result['section']}\n"
            if result.get("program"):
                msg += f"Program:       {result['program']}\n"
            if result["description"]:
                msg += f"\nDescription:\n{result['description'][:400]}"
        else:
            msg = f" {result.get('message', 'Course not found')}"

        dispatcher.utter_message(text=_lang_wrap(msg, lang))
        return []


class ActionGetPrograms(Action):
    def name(self): return "action_get_programs"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        programs = DATA.get("programs") or fetch_api_data("programs")
        if not programs:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        grouped: Dict[str, List[str]] = {}
        for p in programs:
            grouped.setdefault(p.get("degree_type", "Other"), []).append(p.get("name", "Unknown Program"))

        icons = {"Undergraduate": "", "Postgraduate": "", "Graduate": ""}
        message = " East West University Programs:\n\n"
        for dtype, names in grouped.items():
            message += f"{icons.get(dtype,'#')} {dtype}:\n"
            message += "".join(f"   • {n}\n" for n in names) + "\n"

        dispatcher.utter_message(text=_lang_wrap(message.strip(), lang))
        return []


class ActionGetGradingSystem(Action):
    def name(self): return "action_get_grading_system"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        raw = DATA.get("grade_scale") or fetch_api_data("grade-scale")
        if not isinstance(raw, list) or not raw:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        regular = [g for g in raw if not g.get("is_special", False)]
        special  = [g for g in raw if g.get("is_special", False)]

        message = " EWU Grading System\n\nGrade Scale:\n"
        for g in regular:
            try:
                message += f"  {g.get('letter_grade',''):<3} {float(g.get('grade_point',0)):.2f}  ({g.get('numerical_score','')})\n"
            except Exception:
                message += f"  {g.get('letter_grade',''):<3} {g.get('grade_point','')}  ({g.get('numerical_score','')})\n"

        if special:
            message += "\nSpecial Grades:\n"
            for g in special:
                message += f"  {g.get('letter_grade','')}: {g.get('description','')}\n"
                if g.get("note"):
                    message += f"    * {g.get('note')}\n"

        dispatcher.utter_message(text=_lang_wrap(message, lang))
        return []


class ActionGetTuitionFees(Action):
    def name(self):
        return "action_get_tuition_fees"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        fees_list = DATA.get("tuition_fees") or fetch_api_data("tuition-fees")
        if not isinstance(fees_list, list) or not fees_list:
            dispatcher.utter_message(
                text=_lang_wrap(
                    "Tuition fee data unavailable right now. "
                    "Visit https://www.ewubd.edu or contact accounts@ewubd.edu.",
                    lang,
                )
            )
            return []

        def _fmt(val):
            try:
                return f"{int(float(val)):,}"
            except Exception:
                return str(val) if val is not None else "N/A"

        def _fee_row(p):
            currency = p.get("currency", "BDT")
            row  = f"\n   {p.get('program', 'N/A')}\n"
            row += f"     Total Credits          : {p.get('credits', 'N/A')}\n"
            row += f"     Per Credit Fee         : {_fmt(p.get('fee_per_credit'))} {currency}\n"
            row += f"     Tuition Fees           : {_fmt(p.get('total_tuition'))} {currency}\n"
            row += f"     Library/Lab/Activities : {_fmt(p.get('library_lab_fees'))} {currency}\n"
            row += f"     Admission Fee          : {_fmt(p.get('admission_fee'))} {currency}\n"
            row += f"     \n"
            row += f"      GRAND TOTAL         : {_fmt(p.get('grand_total'))} {currency}\n"
            return row

        ug_rows   = [r for r in fees_list if r.get("level", "").lower() == "undergraduate"]
        grad_rows = [r for r in fees_list if r.get("level", "").lower() in ("graduate", "postgraduate")]

        msg = " EWU Tuition Fees — All Programs:\n"

        if ug_rows:
            msg += "\n Undergraduate Programs (Per Credit):\n"
            for row in ug_rows:
                msg += _fee_row(row)

        if grad_rows:
            msg += "\n Graduate Programs (Per Credit):\n"
            for row in grad_rows:
                msg += _fee_row(row)

        dispatcher.utter_message(text=_lang_wrap(msg, lang))
        return []

class ActionGetScholarships(Action):
    def name(self): return "action_get_scholarships"

    _GITHUB_FILE = "scholarships_and_financial_aids.json"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        raw = load_from_github(self._GITHUB_FILE)
        if not raw or not isinstance(raw, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        msg = "EWU Scholarships & Financial Assistance\n\n"

        #  Merit Scholarships 
        merit = raw.get("merit_scholarships", {})
        if merit:
            msg += " Merit Scholarships:\n"
            for key, details in merit.items():
                if not isinstance(details, dict):
                    continue
                title = key.replace("_", " ").title()
                msg += f"\n  • {title}\n"
                for field in ("scholarship", "duration", "eligibility", "minimum_score"):
                    val = details.get(field)
                    if val:
                        msg += f"    {field.replace('_',' ').title()}: {val}\n"
            msg += "\n"

        #  Financial Assistance 
        fa = raw.get("financial_assistance", {})
        if fa:
            msg += "Financial Assistance:\n"
            for key, details in fa.items():
                if not isinstance(details, dict):
                    continue
                title = key.replace("_", " ").title()
                msg += f"\n  • {title}\n"
                for field in ("quota", "scholarship", "minimum_cgpa", "benefit", "eligibility"):
                    val = details.get(field)
                    if val:
                        msg += f"    {field.replace('_',' ').title()}: {val}\n"
            msg += "\n"

        #  Graduate Scholarship Slabs 
        grad_req = raw.get("graduate_scholarship_requirements", {})
        slabs = grad_req.get("scholarships", [])
        if slabs:
            msg += "Graduate Scholarships (Based on Bachelor's CGPA):\n"
            for s in slabs:
                msg += f"  • CGPA {s.get('bachelor_cgpa','N/A')}: {s.get('scholarship_percentage','N/A')} tuition waiver\n"
            msg += "\n"

        #  Scholarship Types 
        types = raw.get("scholarship_types", [])
        if types:
            msg += "Scholarship Types:\n"
            for t in types:
                msg += f"  • {t}\n"
            msg += "\n"

        #  Overview numbers 
        overview = raw.get("overview", {})
        if overview:
            msg += (
                f"Overview:\n"
                f"  • Students benefiting: {overview.get('percentage_of_students_benefiting','N/A')}\n"
                f"  • Earnings distributed as scholarship: {overview.get('percentage_of_earnings_distributed','N/A')}\n\n"
            )

        msg += " Contact: admissions@ewubd.edu |  09666775577"

        dispatcher.utter_message(text=_lang_wrap(msg, lang))
        return []

class ActionGetHelpdeskContacts(Action):
    def name(self): return "action_get_helpdesk_contacts"

    _GITHUB_FILE = "helpdesk_contacts.json"

    _ADMIN_ALIASES = {
        "register":    "registrar",
        "registrar":   "registrar",
        "reg office":  "registrar",
        "account":     "accounts",
        "finance":     "accounts",
        "financial":   "accounts",
        "it":          "ics",
        "tech":        "ics",
        "certificate": "coe",
        "transcript":  "coe",
        "grade report":"coe",
    }

    #  internal helpers 

    def _load_raw(self) -> dict:
        """Return the helpdesk JSON dict, preferring preloaded DATA."""
        raw = DATA.get("helpdesk_contacts") or load_from_github(self._GITHUB_FILE)
        return raw if isinstance(raw, dict) else {}

    def _flatten_entries(self, raw: dict) -> list:
        """
        Convert the nested JSON into a flat list of normalised entry dicts:
          {
            category     : "academic" | "administrative",
            code         : str,   # dept code or office name
            full_name    : str,
            email        : str,
            purpose      : str,
          }
        """
        entries = []
        helpdesks = raw.get("department_helpdesks", {})

        for val in helpdesks.get("academic_departments", {}).values():
            if not isinstance(val, dict):
                continue
            entries.append({
                "category":  "academic",
                "code":      val.get("department", "").strip(),
                "full_name": val.get("full_name", "").strip(),
                "email":     val.get("email", "").strip(),
                "purpose":   "",
            })

        for key, val in helpdesks.get("administrative_offices", {}).items():
            if not isinstance(val, dict):
                continue
            entries.append({
                "category":  "administrative",
                "code":      val.get("office", key).strip(),
                "full_name": val.get("office", key).strip(),
                "email":     val.get("email", "").strip(),
                "purpose":   val.get("purpose", "").strip(),
                "_key":      key.lower(),   
            })

        return entries

    def _find_match(self, user_message: str, entries: list):
        import re

        msg = user_message.lower()

        for entry in entries:
            full_name_lower = entry["full_name"].lower()

          
            if full_name_lower and full_name_lower in msg:
                return entry

       
            code_lower = entry["code"].lower()
            if code_lower and re.search(r"\b" + re.escape(code_lower) + r"\b", msg):
                return entry

     
        for alias, target_key in self._ADMIN_ALIASES.items():
            if alias in msg:
                for entry in entries:
                    if entry.get("_key") == target_key or entry["code"].lower() == target_key:
                        return entry

        return None



    def run(self, dispatcher, tracker, domain):
        lang         = get_user_language(tracker)
        user_message = tracker.latest_message.get("text", "").lower()
        logger.info(f"[action_get_helpdesk_contacts] lang={lang}")

        raw = self._load_raw()
        if not raw:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        entries = self._flatten_entries(raw)
        if not entries:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

       
        matched = self._find_match(user_message, entries)
        if matched:
            msg  = f"EWU Helpdesk - {matched['full_name']}\n\n"
            msg += f"Dept/Office : {matched['code']}\n"
            msg += f"Email       : {matched['email']}\n"
            if matched.get("purpose"):
                msg += f"Purpose     : {matched['purpose']}\n"
            dispatcher.utter_message(text=_lang_wrap(msg, lang))
            return []

     
        academic = [e for e in entries if e["category"] == "academic"]
        admin    = [e for e in entries if e["category"] == "administrative"]

        msg = "EWU Helpdesk Contacts\n\nAcademic Departments:\n"
        for e in academic:
            msg += f"   - {e['full_name']} ({e['code']})\n"
            msg += f"     Email: {e['email']}\n"

        msg += "\nAdministrative Offices:\n"
        for e in admin:
            msg += f"   - {e['full_name']} ({e['code']})\n"
            msg += f"     Email: {e['email']}\n"
            if e.get("purpose"):
                msg += f"     Purpose: {e['purpose']}\n"

        msg += (
            "\nFor general enquiries:\n"
            "   https://www.ewubd.edu | admissions@ewubd.edu | 09666775577"
        )
        dispatcher.utter_message(text=_lang_wrap(msg, lang))
        return []




class ActionGetAdmissionDeadlines(Action):
    def name(self): return "action_get_admission_deadlines"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        raw_list = DATA.get("admission_deadlines") or fetch_api_data("admission-deadlines")
        if not raw_list:
            dispatcher.utter_message(
                text=_lang_wrap(
                    "Admission deadline info unavailable. "
                    "Contact admissions@ewubd.edu or call 09666775577.",
                    lang,
                )
            )
            return []

        programs = [p for p in raw_list if isinstance(p, dict)]
        if not programs:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        user_message = tracker.latest_message.get("text", "").lower()

        ug_list   = [p for p in programs if p.get("level", "").lower() == "undergraduate"]
        grad_list = [p for p in programs if p.get("level", "").lower() in ("graduate", "postgraduate")]

        def has_keyword(text, keywords):
            return any(re.search(r'\b' + re.escape(kw.strip()) + r'\b', text) for kw in keywords)
        
        wants_grad = has_keyword(user_message, [
            "masters", "master", "graduate", "postgraduate", "ms", "mba", "emba", "mss", "ma"
        ])
        wants_ug = has_keyword(user_message, [
            "undergraduate", "bsc", "bba", "ba", "llb", "bachelor"
        ])

        pool = grad_list if wants_grad else (ug_list if wants_ug else ug_list + grad_list)

        def match_program(p):
            haystack = " ".join([
                p.get("program", ""),
                p.get("department", ""),
                p.get("department_code", ""),
            ]).lower()
            tokens = [t for t in user_message.split() if len(t) >= 2]
            return any(tok in haystack for tok in tokens)

        matched     = [p for p in pool if match_program(p)]
        sem         = programs[0].get("semester", "")
        message     = f"EWU Admission Deadlines{' — ' + sem if sem else ''}\n\n"
        render_list = matched if matched else (pool or programs)

        if matched or len(render_list) <= 30:
            for p in render_list:
                name = p.get("program") or p.get("department") or "N/A"
                message += f" {name}\n"
                if p.get("application_deadline"):
                    message += f"    Application Deadline : {p['application_deadline']}\n"
                if p.get("admission_test_date"):
                    message += f"    Admission Test       : {p['admission_test_date']}\n"
                if p.get("result_date"):
                    message += f"    Result               : {p['result_date']}\n"
                message += "\n"
        else:
            if ug_list:
                message += " Undergraduate:\n"
                for p in ug_list:
                    name = p.get("program") or p.get("department") or "N/A"
                    dl   = p.get("application_deadline", "N/A")
                    message += f"   • {name} — {dl}\n"
                message += "\n"
            if grad_list:
                message += " Graduate:\n"
                for p in grad_list:
                    name = p.get("program") or p.get("department") or "N/A"
                    dl   = p.get("application_deadline", "N/A")
                    message += f"   • {name} — {dl}\n"

        dispatcher.utter_message(text=_lang_wrap(message.strip(), lang))
        return []


class ActionGetAdmissionRequirements(Action):
    def name(self): return "action_get_admission_requirements"

    def run(self, dispatcher, tracker, domain):
        data = DATA.get("admission_requirements") or load_from_github(
            GITHUB_DATA_SOURCES.get("admission_requirements", "dynamic_admission_requirements.json")
        )
        if not data:
            utter_smart(dispatcher, tracker, text="Admission requirement information is unavailable.")
            return []

        user_message = tracker.latest_message.get("text", "").lower()
        reqs    = data.get("admission_requirements", {})
        message = " EWU Admission Requirements\n\n"

        wants_graduate = any(kw in user_message for kw in ["masters","master","ms","mba","graduate","emba"])

        dept_keywords = {
            "cse": "cse", "ice": "ice", "eee": "eee",
            "civil": "civil", "geb": "geb",
            "pharmacy": "bpharm", "math": "mathematics",
            "data science": "data_science",
        }
        dept_key = None
        for kw, key in dept_keywords.items():
            if kw in user_message:
                dept_key = key
                break

        #  graduate
        if wants_graduate:
            grad = reqs.get("graduate", {}).get("mba_emba", {})
            message += " Graduate (Masters / MS / MBA) Requirements:\n"
            message += f"  Bachelor's Degree: {grad.get('degree')}\n"
            message += f"  SSC & HSC GPA: {grad.get('ssc_hsc_gpa')}\n"
            message += "  Admission Test: Required (exemptions available)\n"
            if "emba" in user_message:
                message += f"  EMBA Experience: {grad.get('emba', {}).get('work_experience')}\n"
            utter_smart(dispatcher, tracker, text=message)
            return []

        #  undergraduate 
        ug = reqs.get("undergraduate", {}).get("general_programs_except_bpharm", {})
        if dept_key and "subject_requirements" in ug:
            subject_req = ug["subject_requirements"].get(dept_key)
            if subject_req:
                message += f" Undergraduate – {dept_key.upper()} Requirements:\n"
                message += f"  Subject Requirement: {subject_req}\n"
                message += f"  SSC & HSC: {ug.get('ssc_hsc')}\n"
                message += f"  Admission Test Weight: {ug.get('admission_test', {}).get('weightage')}\n"
                utter_smart(dispatcher, tracker, text=message)
                return []

        #  show all
        message += " Undergraduate (General):\n"
        message += f"  SSC & HSC: {ug.get('ssc_hsc')}\n"
        message += f"  Diploma: {ug.get('diploma')}\n"
        message += "  O & A Level: Passed O-Level (5) & A-Level (2)\n\n"

        grad = reqs.get("graduate", {}).get("mba_emba", {})
        message += " Graduate:\n"
        message += "  Bachelor's Degree Required\n"
        message += f"  Min SSC & HSC GPA: {grad.get('ssc_hsc_gpa')}\n\n"

        message += " Required Documents:\n"
        for doc_item in reqs.get("required_documents", []):
            message += f"  • {doc_item}\n"

        utter_smart(dispatcher, tracker, text=message)
        return []


class ActionGetConductRules(Action):
    def name(self): return "action_get_conduct_rules"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        doc = _get_document("student-rules")
        if not doc or not isinstance(doc, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        r   = doc.get("content", {})
        msg = " EWU General Conduct Rules\n\n Expected Behavior:\n"
        for rule in r.get("general_conduct_rules", {}).get("expected_behavior", []):
            msg += f"  • {rule}\n"
        msg += "\n Academic Misconduct (Zero Tolerance):\n"
        for ex in r.get("academic_misconduct", {}).get("examples", []):
            msg += f"  • {ex}\n"
        msg += "\n Social Misconduct:\n"
        for ex in r.get("social_misconduct", {}).get("examples", []):
            msg += f"  • {ex}\n"

        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionGetClubsSocieties(Action):
    def name(self): return "action_get_clubs_societies"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        clubs_raw = DATA.get("clubs") or fetch_api_data("clubs")
        clubs = (
            clubs_raw if isinstance(clubs_raw, list)
            else clubs_raw.get("clubs", []) if isinstance(clubs_raw, dict)
            else []
        )
        if not clubs:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        message = " EWU Clubs and Societies:\n\n"
        for i, club in enumerate(clubs, start=1):
            message += f"{i}. {club.get('name','Unknown Club')}\n"
            if club.get("description"):
                message += f"    {club.get('description')}\n"
            if club.get("url"):
                message += f"    {club.get('url')}\n"
            message += "\n"

        dispatcher.utter_message(text=_lang_wrap(message, lang))
        return []


class ActionGetAlumniInfo(Action):
    def name(self): return "action_get_alumni_info"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        alumni_list = DATA.get("alumni") or fetch_api_data("alumni")
        if not alumni_list:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        message = f" EWU Alumni Network\n\nNotable Alumni: {len(alumni_list)}\n\n"
        for alum in alumni_list:
            if not isinstance(alum, dict):
                continue
            message += f"• {alum.get('name','Unknown')}"
            if alum.get("department"):
                message += f" ({alum.get('department')})"
            message += "\n"
            if alum.get("achievement"):
                message += f"  Achievement: {alum.get('achievement')}\n"
            pos_co = " / ".join(filter(None, [alum.get("position",""), alum.get("company","")]))
            if pos_co:
                message += f"  Position: {pos_co}\n"
            message += "\n"

        dispatcher.utter_message(text=_lang_wrap(message, lang))
        return []


class ActionGetEvents(Action):
    def name(self): return "action_get_events"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        events = DATA.get("events") or fetch_api_data("events")
        if not events:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        message = " EWU Recent Events:\n\n"
        for ev in events[:10]:
            message += f"🔹 {ev.get('title','N/A')}\n"
            if ev.get("event_date"):
                message += f"    {ev.get('event_date')}\n"
            if ev.get("location"):
                message += f"    {ev.get('location')}\n"
            message += "\n"

        dispatcher.utter_message(text=_lang_wrap(message.strip(), lang))
        return []


class ActionGetNotices(Action):
    def name(self): return "action_get_notices"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        notices = DATA.get("notices") or fetch_api_data("notices")
        if not notices:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        message = " EWU Latest Notices:\n\n"
        for n in notices[:10]:
            message += f"• {n.get('title','N/A')}\n"
            if n.get("published_date"):
                message += f"   {n.get('published_date')}\n"
            if n.get("url"):
                message += f"   {n.get('url')}\n"
            message += "\n"

        dispatcher.utter_message(text=_lang_wrap(message.strip(), lang))
        return []


class ActionGetGovernance(Action):
    def name(self): return "action_get_governance"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        user_message = tracker.latest_message.get("text", "").lower()

        body_map = {
            "board of trustees": "board_of_trustees",
            "board":             "board_of_trustees",
            "syndicate":         "syndicate",
            "academic council":  "academic_council",
            "academic":          "academic_council",
        }
        body_filter = next((v for k, v in body_map.items() if k in user_message), None)

        all_members = DATA.get("governance") or fetch_api_data("governance")
        if not all_members:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        
        members = (
            [m for m in all_members if m.get("body") == body_filter]
            if body_filter else all_members
        )
        if not members:
            members = all_members  

        grouped: Dict[str, List] = {}
        for m in members:
            grouped.setdefault(m.get("body", "Other"), []).append(m)

        msg = " EWU Governance:\n\n"
        for body, bm in grouped.items():
            msg += f" {body.replace('_',' ').title()}:\n"
            for m in bm:
                msg += (
                    f"  • {m.get('name','N/A')} — {m.get('role','N/A')}"
                    f"{'  (Chairperson)' if m.get('is_chairperson') else ''}\n"
                )
            msg += "\n"

        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionGetPartnerships(Action):
    def name(self): return "action_get_partnerships"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        partnerships = DATA.get("partnerships") or fetch_api_data("partnerships")
        if not partnerships:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        msg = " EWU International Partnerships:\n\n"
        for p in partnerships:
            msg += f" {p.get('name','N/A')}"
            if p.get("acronym"):
                msg += f" ({p.get('acronym')})"
            msg += "\n"
            if p.get("country"):
                msg += f"   Country: {p.get('country')}\n"
            if p.get("partnership_type"):
                msg += f"   Partnership: {p.get('partnership_type')}\n"
            msg += "\n"

        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionGetNewsletters(Action):
    def name(self): return "action_get_newsletters"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        newsletters = DATA.get("newsletters") or fetch_api_data("newsletters")
        if not newsletters:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        msg = " EWU Newsletters:\n\n"
        for n in newsletters[:10]:
            msg += f"• {n.get('title','N/A')}"
            if n.get("semester"):
                msg += f" — {n.get('semester')}"
            if n.get("year"):
                msg += f" ({n.get('year')})"
            msg += "\n"
            if n.get("pdf_url"):
                msg += f"   {n.get('pdf_url')}\n"
            msg += "\n"

        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionGetProctorSchedule(Action):
    def name(self): return "action_get_proctor_schedule"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)

        schedule = DATA.get("proctor_schedule") or fetch_api_data("proctor-schedule")
        if not schedule:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        grouped: Dict[str, List] = {}
        for entry in schedule:
            grouped.setdefault(entry.get("day_of_week", "Unknown"), []).append(entry)

        day_order = ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"]
        msg = " EWU Proctor Schedule:\n\n"
        for day in day_order:
            if day in grouped:
                msg += f" {day}:\n"
                for e in grouped[day]:
                    msg += (
                        f"  • {e.get('name','N/A')} "
                        f"({e.get('designation','N/A')}, {e.get('department','N/A')})\n"
                    )
                msg += "\n"

        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionGetAcademicCalendar(Action):
    def name(self): return "action_get_academic_calendar"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        user_message = tracker.latest_message.get("text", "").lower()

        wants_exam = "exam" in user_message
        calendar_type = "exam_schedule" if wants_exam else "academic_calendar"

       
        all_calendar = DATA.get("academic_calendar") or []
        if isinstance(all_calendar, list) and all_calendar:
            filtered = [e for e in all_calendar if e.get("calendar_type") == calendar_type]
            calendar = filtered if filtered else all_calendar
        else:
            calendar = fetch_api_data(
                "academic-calendar", params={"calendar_type": calendar_type}
            )

        if not calendar:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        grouped: Dict[str, List] = {}
        for entry in calendar:
            grouped.setdefault(entry.get("semester", "Unknown Semester"), []).append(entry)

        label = "Exam Schedule" if wants_exam else "Academic Calendar"
        msg   = f" EWU {label}:\n\n"
        for sem, events in list(grouped.items())[:3]:
            msg += f"🗓 {sem}:\n"
            for ev in events[:12]:
                msg += f"  • {ev.get('event_date','N/A')} ({ev.get('day','')}) — {ev.get('event_name','N/A')}\n"
            if len(events) > 12:
                msg += f"  … and {len(events)-12} more events\n"
            msg += "\n"

        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionAdmissionApplicationStep(Action):
    def name(self): return "action_admission_application_steps"

    def run(self, dispatcher, tracker, domain):
        process_data = DATA.get("admission_process") or load_from_github(
            GITHUB_DATA_SOURCES.get("admission_process", "dynamic_admission_process.json")
        )

        user_message  = tracker.latest_message.get("text", "").lower()
        wants_graduate = any(kw in user_message for kw in [
            "masters","master","ms","mba","emba","graduate","postgraduate"
        ])

        step_keywords = {
            "document": ["document","paper","certificate","required","photo","photograph","signature","scan"],
            "test":     ["test","exam","admission test","written","admit card","admit"],
            "apply":    ["apply","application","form","register","online","submit","fill"],
            "fee":      ["fee","payment","cost","pay","bkash","tk","taka"],
            "login":    ["login","sign in","id","login id","ewu id","account"],
            "result":   ["result","merit","selected","outcome","qualified"],
        }
        asked_step = None
        for step_key, keywords in step_keywords.items():
            if any(kw in user_message for kw in keywords):
                asked_step = step_key
                break

        if process_data and isinstance(process_data, dict):
            level_key = "graduate" if wants_graduate else "undergraduate"
            steps = (
                process_data.get(level_key, {}).get("application_steps")
                or process_data.get("application_steps")
                or process_data.get("steps")
                or []
            )
            if not steps and isinstance(process_data, list):
                steps = process_data

            if steps:
                level_label = "Graduate" if wants_graduate else "Undergraduate"
                message = f" EWU {level_label} Admission — Application Steps\n\n"
                shown_any = False

                for i, step in enumerate(steps, start=1):
                    if isinstance(step, dict):
                        title    = step.get("step") or step.get("title") or step.get("name") or f"Step {i}"
                        detail   = step.get("description") or step.get("details") or step.get("info") or ""
                        deadline = step.get("deadline") or step.get("date") or ""
                        note     = step.get("note") or step.get("notes") or ""

                        if asked_step:
                            combined = f"{title} {detail}".lower()
                            if not any(kw in combined for kw in step_keywords[asked_step]):
                                continue

                        shown_any  = True
                        message   += f"{'—' * 30}\n"
                        message   += f"Step {i}: {title}\n"
                        if detail:    message += f"   {detail}\n"
                        if deadline:  message += f"   Deadline: {deadline}\n"
                        if note:      message += f"   Note: {note}\n"
                        message   += "\n"
                    else:
                        shown_any  = True
                        message   += f"Step {i}: {step}\n\n"

                if shown_any:
                    contact = process_data.get("contact") or process_data.get("contact_information") or {}
                    if contact:
                        message += " Admission Office:\n"
                        if contact.get("phone"):   message += f"   Phone : {contact['phone']}\n"
                        if contact.get("email"):   message += f"   Email : {contact['email']}\n"
                        if contact.get("website"): message += f"   Web   : {contact['website']}\n"
                    else:
                        message += (
                            " Admission Office:\n"
                            "   Phone : 55046678, 09666775577 | Mobile: 01755587224\n"
                            "   Email : admissions@ewubd.edu\n"
                            "   Web   : http://admission.ewubd.edu\n"
                        )
                    utter_smart(dispatcher, tracker, text=message.strip())
                    return []

        logger.warning(
            "[ActionAdmissionApplicationStep] Could not load admission_process data. "
            "Using official EWU hardcoded guidelines."
        )

        TOPIC_RESPONSES = {
            "fee": (
                " EWU Admission Application Fee\n\n"
                "• Application Fee: Tk. 1,500/- (Non-Refundable)\n"
                "• Pay in cash at the EWU Admission Office\n"
                "• OR pay online via bKash (online processing fee will be charged)\n\n"
                " Admission Office:\n"
                "   Address : A/2 Jahurul Islam Avenue, Jahurul Islam City, Aftabnagar, Dhaka\n"
                "   Phone   : 55046678, 09666775577 | Mobile: 01755587224\n"
                "   Email   : admissions@ewubd.edu"
            ),
            "document": (
                " EWU Admission — Required Documents and Uploads\n\n"
                "1. Passport-size Colored Photograph\n"
                "   • Recent (not older than 6 months)\n"
                "   • Format: .jpg | Max size: 100 KB\n\n"
                "2. Scanned Signature\n"
                "   • Background must be white\n"
                "   • Format: .jpg | Max size: 60 KB\n\n"
                "3. During Admission Test and Admission (if qualified):\n"
                "   • Original HSC / Equivalent Registration Card\n"
                "   • OR A-Level Statement of Entry\n\n"
                " Note: If any false or wrong information is detected at any time, "
                "the application and admission (if qualified) will be cancelled.\n\n"
                " Admission Office: 55046678, 09666775577 | admissions@ewubd.edu"
            ),
            "login": (
                " EWU Admission — Login and EWU Login ID\n\n"
                "1. Go to http://admission.ewubd.edu\n"
                "2. Click 'Create New Applicant' and select your program\n"
                "3. Fill in your Name, Mobile Number and Email (optional)\n"
                "4. Your EWU Login ID will be displayed on the next screen\n"
                "   Example: 163354131\n"
                "   • Note it down — you will need it for payment and future logins\n"
                "   • The Login ID is also sent to your email\n"
                "5. After payment, sign in using your EWU Login ID + Mobile Number\n\n"
                " Need help? Call: 55046678, 09666775577 | admissions@ewubd.edu"
            ),
            "test": (
                " EWU Admission Test — Admit Card and Instructions\n\n"
                "• After submitting your Online Admission Form, download and print your Admit Card\n"
                "• Log in any time using your EWU Login ID + Mobile Number to print the Admit Card\n"
                "• Without the Admit Card you will NOT be allowed to sit the Admission Test\n\n"
                "On the day of the test, bring:\n"
                "   • Printed Admit Card\n"
                "   • Original HSC / Equivalent Registration Card\n"
                "   • OR A-Level Statement of Entry\n\n"
                "The decision of the EWU Admission Committee is final.\n\n"
                " Admission Office: 55046678, 09666775577 | admissions@ewubd.edu"
            ),
            "apply": (
                " EWU Admission — How to Fill and Submit the Online Form\n\n"
                "1. Go to http://admission.ewubd.edu\n"
                "2. Click 'Create New Applicant' then select your program\n"
                "   • After selecting a program you cannot change it\n"
                "3. Fill in Name, Mobile Number, Email (optional) then note your EWU Login ID\n"
                "4. Pay the Application Fee (Tk. 1,500/- cash or bKash)\n"
                "5. Sign in with EWU Login ID + Mobile Number\n"
                "6. Complete all fields marked with (*) — mandatory fields\n"
                "7. Upload Photograph (.jpg, max 100 KB) and Signature (.jpg, max 60 KB)\n"
                "8. Use <Save> to save progress, <Preview> to review, <Submit Form> to submit\n"
                "   • After submission the form cannot be edited\n"
                "9. On success you will see: 'Your form has been submitted successfully.'\n"
                "   Then click 'Admit Card' to download it\n"
                "10. Print your Admit Card — required to appear in the Admission Test\n\n"
                " Admission Office: 55046678, 09666775577 | admissions@ewubd.edu"
            ),
        }

        if asked_step and asked_step in TOPIC_RESPONSES:
            utter_smart(dispatcher, tracker, text=TOPIC_RESPONSES[asked_step])
            return []

        steps = [
            (
                "Access the Online Admission Form",
                "Open any web browser (Google Chrome or Mozilla Firefox recommended).\n"
                "   Go to http://www.ewubd.edu OR directly to http://admission.ewubd.edu"
            ),
            (
                "Create a New Applicant Account and Select Program",
                "Click 'Create New Applicant', then select your desired program by clicking 'Click here to apply'.\n"
                "   After selecting a program you cannot change it."
            ),
            (
                "Get Your EWU Login ID",
                "Fill in your Name, Mobile Number, and Email (optional).\n"
                "   Your EWU Login ID (e.g. 163354131) will appear on the next screen.\n"
                "   Note it down — you need it for payment and future logins.\n"
                "   The Login ID is also sent to your email."
            ),
            (
                "Pay the Application Fee",
                "Application Fee: Tk. 1,500/- (Non-Refundable)\n"
                "   • Pay in cash at the EWU Admission Office\n"
                "   • OR pay online via bKash (online processing fee applies)"
            ),
            (
                "Sign In and Fill the Online Admission Form",
                "After payment, sign in at http://admission.ewubd.edu using your EWU Login ID + Mobile Number.\n"
                "   • Fill all fields carefully — fields marked (*) are mandatory\n"
                "   • You are responsible for the correctness of your information\n"
                "   • Upload Photograph (.jpg, max 100 KB — taken within 6 months)\n"
                "   • Upload Signature (.jpg, max 60 KB — white background)"
            ),
            (
                "Save, Preview and Submit",
                "Use the buttons at the bottom of the form:\n"
                "   <Save>         — Save for later editing\n"
                "   <Preview>      — Review and download as PDF (strongly recommended before submitting)\n"
                "   <Submit Form>  — Final submission\n"
                "   • Once submitted, the form cannot be edited."
            ),
            (
                "Download Your Admit Card",
                "After successful submission you will see: 'Your form has been submitted successfully. "
                "For obtaining the Admit Card click on the Admit Card.'\n"
                "   • Click 'Admit Card' to download and print it\n"
                "   • You can reprint any time by logging in with your EWU Login ID + Mobile Number\n"
                "   • Without the Admit Card you will NOT be allowed to sit the Admission Test"
            ),
            (
                "Appear for the Admission Test",
                "Bring to the test:\n"
                "   • Printed Admit Card\n"
                "   • Original HSC / Equivalent Registration Card OR A-Level Statement of Entry\n"
                "   • The decision of the EWU Admission Committee is final."
            ),
        ]

        message = " EWU Online Admission — Step-by-Step Guide\n\n"
        for i, (title, detail) in enumerate(steps, start=1):
            message += f"{'—' * 32}\n"
            message += f"Step {i}: {title}\n"
            message += f"   {detail}\n\n"

        message += (
            " Admission Office\n"
            "   Address : A/2 Jahurul Islam Avenue, Jahurul Islam City, Aftabnagar, Dhaka\n"
            "   Phone   : 55046678, 09666775577\n"
            "   Mobile  : 01755587224\n"
            "   Email   : admissions@ewubd.edu\n"
            "   Web     : http://admission.ewubd.edu\n\n"
            "Tip: Ask me about a specific step — e.g. 'How do I pay the fee?' or 'What documents do I need?'"
        )

        utter_smart(dispatcher, tracker, text=message.strip())
        return []




class ActionGetProgramDetails(Action):
    def name(self): return "action_get_program_details"

   
    _SECTIONS = [
        ("undergraduate_programs", " Undergraduate Programs", "UG"),
        ("graduate_programs",      " Graduate Programs",      "Graduate"),
        ("Diploma Programs",       " Diploma Programs",       "Diploma"),
    ]

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        logger.info(f"[action_get_program_details] lang={lang}")

        
        raw = DATA.get("static_programs") or load_from_github(
            GITHUB_DATA_SOURCES["static_programs"]
        )

        if not raw or not isinstance(raw, dict):
            dispatcher.utter_message(
                text=_lang_wrap(
                    "Program credit information is unavailable right now. "
                    "Visit https://www.ewubd.edu or contact admissions@ewubd.edu.",
                    lang,
                )
            )
            return []

     
        user_msg   = tracker.latest_message.get("text", "").lower()
        wants_grad = any(kw in user_msg for kw in [
            "graduate", "masters", "master", "mba", "ms", "postgraduate",
        ])
        wants_ug   = any(kw in user_msg for kw in [
            "undergraduate", "bachelor", "bsc", "bba",
        ])
        wants_dipl = any(kw in user_msg for kw in ["diploma", "pgd", "ppdm"])

        msg = " EWU Programs — Total Credits\n\n"
        has_content = False

        for json_key, section_label, level_tag in self._SECTIONS:
           
            if wants_grad  and level_tag not in ("Graduate",):
                continue
            if wants_ug    and level_tag != "UG":
                continue
            if wants_dipl  and level_tag != "Diploma":
                continue

            section_data = raw.get(json_key)
            if not section_data or not isinstance(section_data, dict):
                continue

            section_lines = []
            for prog_name, prog_info in section_data.items():
                if not isinstance(prog_info, dict):
                    continue
                
                credits = (
                    prog_info.get("total_credits")
                    or prog_info.get("total_credit_hours")
                    or "N/A"
                )
                section_lines.append(f"   • {prog_name}\n     Total Credits : {credits}")

            if section_lines:
                has_content = True
                msg += f"{section_label}:\n"
                msg += "\n".join(section_lines)
                msg += "\n\n"

        if not has_content:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        msg += " For more details: admissions@ewubd.edu |  09666775577"
        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


class ActionGetFacilities(Action):

    def name(self): return "action_get_facilities"

    def run(self, dispatcher, tracker, domain):
        lang = get_user_language(tracker)
        logger.info(f"[action_get_facilities] lang={lang}")

        
        raw = DATA.get("facilities") or load_from_github("dynamic_facilites.json")
        if not raw or not isinstance(raw, dict):
            dispatcher.utter_message(
                text=_lang_wrap(
                    "Facilities information is unavailable right now. "
                    "Visit https://www.ewubd.edu for details.",
                    lang,
                )
            )
            return []

    
        facilities_root = raw.get("facilities", raw)
        if not isinstance(facilities_root, dict):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        msg = "EWU Campus Facilities:\n\n"
        has_content = False

    
        campus_life = facilities_root.get("campus_life", {})
        if isinstance(campus_life, dict):
            available     = campus_life.get("available", [])
            not_available = campus_life.get("not_available", [])

            if available:
                has_content = True
                msg += "Campus Life - Available:\n"
                for item in available:
                    if not isinstance(item, dict):
                        continue
                    msg += f"   - {item.get('name', 'N/A')}\n"
                    if item.get("description"):
                        msg += f"     {item['description']}\n"
                    if item.get("url"):
                        msg += f"     {item['url']}\n"
                msg += "\n"

            if not_available:
                has_content = True
                msg += "Campus Life - Not Available:\n"
                for item in not_available:
                    if isinstance(item, dict):
                        msg += f"   - {item.get('name', 'N/A')}: {item.get('description', '')}\n"
                msg += "\n"


        ics = facilities_root.get("ics_services", {})
        if isinstance(ics, dict):
            services = ics.get("services", [])
            if services:
                has_content = True
                msg += f"{ics.get('description', 'ICS Services')}:\n"
                for svc in services:
                    msg += f"   - {svc}\n"
                msg += "\n"

  
        rc = facilities_root.get("research_center", {})
        if isinstance(rc, dict):
            rc_facilities = rc.get("facilities", [])
            if rc_facilities:
                has_content = True
                msg += f"{rc.get('name', 'Research Center')}:\n"
                for fac in rc_facilities:
                    if not isinstance(fac, dict):
                        continue
                    msg += f"   - {fac.get('name', 'N/A')}\n"
                    desc = fac.get("description", "")
                    if desc:
                        msg += f"     {desc[:160].rstrip()}{'...' if len(desc) > 160 else ''}\n"
                    if fac.get("location"):
                        msg += f"     Location: {fac['location']}\n"
                msg += "\n"

 
        eng = facilities_root.get("engineering_labs", {})
        if isinstance(eng, dict):
            labs  = eng.get("labs", [])
            depts = eng.get("departments", [])
            if labs or depts:
                has_content = True
                msg += "Engineering Labs:\n"
                for dept in depts:
                    msg += f"   - {dept}\n"
                if labs:
                    msg += f"   Total labs: {len(labs)}\n"
                    for lab in labs:
                        if isinstance(lab, dict):
                            msg += f"   - {lab.get('name', 'N/A')}\n"
                            if lab.get("description"):
                                msg += f"     {lab['description'][:120]}\n"
                msg += "\n"

  
        pharm = facilities_root.get("pharmacy_labs", {})
        if isinstance(pharm, dict):
            if pharm.get("description") or pharm.get("major_equipment"):
                has_content = True
                msg += "Pharmacy Labs:\n"
                if pharm.get("description"):
                    msg += f"   {pharm['description']}\n"
                for eq in pharm.get("major_equipment", []):
                    if isinstance(eq, dict):
                        msg += f"   - {eq.get('name', 'N/A')}"
                        if eq.get("description"):
                            msg += f": {eq['description'][:80]}"
                        msg += "\n"
                msg += "\n"

    
        civil = facilities_root.get("civil_engineering_labs", {})
        if isinstance(civil, dict):
            labs = civil.get("labs", [])
            if labs:
                has_content = True
                msg += "Civil Engineering Labs:\n"
                for lab in labs:
                    if isinstance(lab, dict):
                        msg += f"   - {lab.get('name', 'N/A')}"
                        if lab.get("location"):
                            msg += f"  (Location: {lab['location']})"
                        msg += "\n"
                        if lab.get("description"):
                            msg += f"     {lab['description'][:100]}\n"
                msg += "\n"

        if not has_content:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        msg += (
            "For more information:\n"
            "   https://www.ewubd.edu | admissions@ewubd.edu | 09666775577"
        )
        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []

class ActionGetEventDetails(Action):
    def name(self): return "action_get_event_details"

    _EVENT_TYPE_KEYWORDS = {
        "seminar":   ["seminar", "সেমিনার"],
        "workshop":  ["workshop", "ওয়ার্কশপ"],
        "fest":      ["fest", "festival", "উৎসব", "ফেস্ট"],
        "competition": ["competition", "contest", "প্রতিযোগিতা"],
        "webinar":   ["webinar", "online event", "ওয়েবিনার"],
        "cultural":  ["cultural", "সাংস্কৃতিক"],
        "sports":    ["sport", "game", "খেলা", "ক্রীড়া"],
    }

    def _detect_type(self, user_message: str) -> str:
        for etype, keywords in self._EVENT_TYPE_KEYWORDS.items():
            if any(kw in user_message for kw in keywords):
                return etype
        return ""

    def run(self, dispatcher, tracker, domain):
        lang         = get_user_language(tracker)
        user_message = tracker.latest_message.get("text", "").lower()
        logger.info(f"[action_get_event_details] lang={lang}")

        events = DATA.get("events") or fetch_api_data("events")
        if not events or not isinstance(events, list):
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        #  optional type filter 
        event_type = self._detect_type(user_message)
        if event_type:
            filtered = [
                ev for ev in events
                if isinstance(ev, dict) and any(
                    event_type in str(ev.get(field, "")).lower()
                    for field in ("title", "event_type", "category", "description")
                )
            ]
            events = filtered if filtered else events

        if not events:
            dispatcher.utter_message(text=_NO_INFO_MSG[lang])
            return []

        label = f" — {event_type.title()}" if event_type else ""
        msg   = f" EWU Upcoming Events{label}:\n\n"

        for ev in events:
            if not isinstance(ev, dict):
                continue
            msg += f"🔹 {ev.get('title', 'N/A')}\n"
            if ev.get("event_date"):
                msg += f"    Date    : {ev.get('event_date')}\n"
            if ev.get("location"):
                msg += f"    Venue   : {ev.get('location')}\n"
            if ev.get("event_type") or ev.get("category"):
                msg += f"   🏷️  Type    : {ev.get('event_type') or ev.get('category')}\n"
            if ev.get("description"):
                desc = ev["description"]
                msg += f"    Details : {desc[:180].rstrip()}{'…' if len(desc) > 180 else ''}\n"
            msg += "\n"

        msg += (
            " For more info on events:\n"
            "   https://www.ewubd.edu |  09666775577"
        )
        dispatcher.utter_message(text=_lang_wrap(msg.strip(), lang))
        return []


# KEYWORD ROUTER + DEFAULT FALLBACK


def _build_router():
    return [
        (ActionGetLocation, frozenset([
            "কোথায়","অবস্থিত","ঠিকানা","কোথা","অবস্থান",
            "kothay","thikana","oboshtit","obosthan",
            "location","address","where is ewu","situated",
        ])),
        (ActionGetEWUHistory, frozenset([
            "ইতিহাস","প্রতিষ্ঠা","প্রতিষ্ঠিত",
            "itihas","protishtit","protistha",
            "history","founded","established",
        ])),
        (ActionGetEWUVision, frozenset([
            "ভিশন","দৃষ্টিভঙ্গি",
            "vision","bhishon",
        ])),
        (ActionGetEWUMission, frozenset([
            "মিশন","লক্ষ্য",
            "mission","lakkho",
        ])),
        (ActionGetTuitionFees, frozenset([
            "খরচ","ফি","বেতন","টাকা","ভর্তি ফি",
            "khoroch","fees","tuition","cost","fee",
        ])),
        (ActionGetScholarships, frozenset([
            "বৃত্তি","স্কলারশিপ",
            "britti","scholarship","waiver",
        ])),
        (ActionAdmissionApplicationStep, frozenset([
            "apply online","application process","application steps","how to apply",
            "online admission","admission form","admit card","login id",
            "ভর্তি আবেদন","আবেদনের ধাপ","কিভাবে আবেদন","অনলাইন ভর্তি",
        ])),
        (ActionGetAdmissionDeadlines, frozenset([
            "ভর্তির সময়সীমা","আবেদনের শেষ তারিখ","ডেডলাইন",
            "deadline","last date","admission date",
        ])),
        (ActionGetAdmissionRequirements, frozenset([
            "ভর্তির যোগ্যতা","ভর্তির শর্ত","ভর্তি হতে",
            "requirements","eligibility","admit",
        ])),
        (ActionGetCourses, frozenset([
            "কোর্স","বিষয়",
            "course","subject","curriculum",
        ])),
        (ActionGetPrograms, frozenset([
            "প্রোগ্রাম","বিভাগ",
            "program","programme","department",
        ])),
        (ActionGetGradingSystem, frozenset([
            "গ্রেড","জিপিএ","সিজিপিএ",
            "grade","gpa","cgpa","marking",
        ])),
        (ActionGetAcademicCalendar, frozenset([
            "একাডেমিক ক্যালেন্ডার","পরীক্ষার সময়সূচি",
            "calendar","exam schedule","academic calendar",
        ])),
        (ActionGetClubsSocieties, frozenset([
            "ক্লাব","সংগঠন",
            "club","society","extracurricular",
        ])),
        (ActionGetAlumniInfo, frozenset([
            "অ্যালামনাই","প্রাক্তন শিক্ষার্থী",
            "alumni","alumnus","graduate student",
        ])),
        (ActionGetEvents, frozenset([
            "ইভেন্ট","অনুষ্ঠান",
            "event","program schedule",
        ])),
        (ActionGetNotices, frozenset([
            "নোটিশ","বিজ্ঞপ্তি",
            "notice","announcement",
        ])),
        (ActionGetGovernance, frozenset([
            "গভর্ন্যান্স","বোর্ড","সিন্ডিকেট","একাডেমিক কাউন্সিল",
            "governance","board of trustees","syndicate","academic council",
        ])),
        (ActionGetHelpdeskContacts, frozenset([
            "হেল্পডেস্ক","যোগাযোগ","ফোন নম্বর",
            "helpdesk","contact","phone number",
        ])),
        (ActionGetPartnerships, frozenset([
            "পার্টনারশিপ","চুক্তি",
            "partnership","collaboration","mou",
        ])),
        (ActionGetProctorSchedule, frozenset([
            "প্রক্টর","শৃঙ্খলা",
            "proctor","disciplin",
        ])),
        (ActionGetSocialMedia, frozenset([
            "সোশ্যাল মিডিয়া","ফেসবুক",
            "social media","facebook","linkedin","instagram",
        ])),
        (ActionGetConductRules, frozenset([
            "নিয়মকানুন","আচরণবিধি",
            "conduct","rules","regulation",
        ])),
        (ActionGetNewsletters, frozenset([
            "নিউজলেটার",
            "newsletter",
        ])),
        (ActionListDepartments, frozenset([
            "বিভাগসমূহ","বিভাগগুলো",
            "departments","faculties",
        ])),
        (ActionGetProgramDetails, frozenset([
            "ক্রেডিট","কত ক্রেডিট","মেয়াদ","সময়কাল",
            "credit","duration","how long","total credits","program duration",
        ])),
        (ActionGetFacilities, frozenset([
            "সুবিধা","ক্যাম্পাস সুবিধা","ল্যাব","লাইব্রেরি","ক্যাফেটেরিয়া","হোস্টেল", "চিকিৎসা",
            "facilities","facility","lab","library","cafeteria","transport","gym","hostel", "medical", "checkup",
        ])),
        (ActionGetEventDetails, frozenset([
            "ইভেন্ট","অনুষ্ঠান","আসন্ন ইভেন্ট","সেমিনার","ওয়ার্কশপ",
            "event","events","upcoming event","seminar","workshop","fest",
        ])),
    ]







import inspect

class ActionDefaultFallback(Action):
    def name(self):
        return "action_default_fallback"

    def __init__(self):
        self.rag_url = "https://atkiya110-rag-server.hf.space/rag"
        self.timeout = 120
        self.max_retries = 1
        self._router = _build_router()

    def _route(self, text: str):
        text_lower = text.lower()
        for action_cls, keywords in self._router:
            for kw in keywords:
                if kw.lower() in text_lower:
                    return action_cls
        return None

    async def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text", "").strip()
        lang = detect_language(user_message) if user_message else "english"
        logger.info(f"[action_default_fallback] lang={lang}  query={user_message!r}")

        if not user_message:
            dispatcher.utter_message(text=_lang_wrap("I didn't catch that. Could you rephrase?", lang))
            return []

        matched_cls = self._route(user_message)
        if matched_cls is not None:
            logger.info(f"[action_default_fallback] keyword-routed to {matched_cls.__name__}")
            result = matched_cls().run(dispatcher, tracker, domain)
            if inspect.isawaitable(result):
                return await result
            return result

   
        try:
            query_embedding = embed_query(user_message)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            query_embedding = None

  
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as session:
                    payload: Dict[str, Any] = {"query": user_message, "top_k": 5}
                    if query_embedding is not None:
                        payload["query_embedding"] = query_embedding

                    async with session.post(self.rag_url, json=payload) as response:
                        if response.status == 200:
                            result     = await response.json()
                            raw_answer = result.get("answer", "").strip()
                            rag_lang   = result.get("detected_language", lang)
                            if rag_lang not in ("bangla", "banglish", "english"):
                                rag_lang = lang

                            if not raw_answer or raw_answer.startswith("["):
                                sources = result.get("sources", [])
                                if sources:
                                    raw_answer = (
                                        f"Found {len(sources)} relevant record(s) in the "
                                        "EWU database but could not generate a summary. "
                                        "Please contact admissions@ewubd.edu."
                                    )
                                else:
                                    dispatcher.utter_message(text=_NO_INFO_MSG[rag_lang])
                                    return []

                            dispatcher.utter_message(text=_lang_wrap(raw_answer, rag_lang))
                            return []

            except asyncio.TimeoutError:
                logger.warning(f"RAG timeout on attempt {attempt + 1}")
            except aiohttp.ClientConnectorError:
                logger.error(f"Cannot connect to RAG at {self.rag_url}")
                break
            except Exception as e:
                logger.error(f"Unexpected RAG error: {e}", exc_info=True)
                if attempt >= self.max_retries - 1:
                    break

        dispatcher.utter_message(text=_CONTACT_MSG[lang])
        return []
