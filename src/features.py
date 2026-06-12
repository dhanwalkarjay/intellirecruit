"""
features.py — Feature extraction from a single candidate dict.

Three responsibilities:
  1. build_candidate_text()  → rich text string to embed semantically
  2. extract_skill_names()   → flat list of skill name strings
  3. get_career_summary()    → short human-readable career string for reasoning
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# 1. Candidate text builder (for semantic embedding)
# ─────────────────────────────────────────────────────────────

def build_candidate_text(candidate: dict) -> str:
    """
    Combines profile, career history, and skills into a single rich paragraph
    that captures what the candidate actually does — not just what they list.

    The career description text is the highest-signal field: it captures
    real systems built, not just skill keywords.
    """
    p       = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    certs   = candidate.get("certifications", [])

    parts: list[str] = []

    # Headline and summary (self-description)
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])

    # Current role context
    current = " | ".join(filter(None, [
        p.get("current_title", ""),
        p.get("current_company", ""),
        p.get("current_industry", ""),
        f"{p.get('years_of_experience', 0):.1f} years experience",
    ]))
    if current:
        parts.append(current)

    # Career descriptions — most important signal for "has built real systems"
    # Take up to 4 most recent roles, first 400 chars of description each
    for job in career[:4]:
        desc = job.get("description", "").strip()
        if desc:
            role_context = f"{job.get('title', '')} at {job.get('company', '')} " \
                           f"({job.get('industry', '')}): {desc[:400]}"
            parts.append(role_context)

    # Skills — with proficiency level (helps semantic matching)
    skill_strs = [
        f"{s['name']} ({s.get('proficiency', '')})"
        for s in skills
        if s.get("name")
    ]
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs))

    # Certifications
    cert_names = [c["name"] for c in certs if c.get("name")]
    if cert_names:
        parts.append("Certifications: " + ", ".join(cert_names))

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
# 2. Skill name extractor
# ─────────────────────────────────────────────────────────────

def extract_skill_names(candidate: dict) -> list[str]:
    """
    Returns a list of skill name strings from the candidate's skills array.
    Also pulls skill names that appear in career descriptions for richer matching.
    """
    skills = candidate.get("skills", [])
    names = [s["name"] for s in skills if s.get("name")]

    # Also include skill names from assessment scores dict (redrob_signals)
    assessed = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for skill_name in assessed:
        if skill_name not in names:
            names.append(skill_name)

    return names


def get_skill_objects(candidate: dict) -> list[dict]:
    """Returns the full skill objects (name, proficiency, endorsements, duration_months)."""
    return candidate.get("skills", [])


def get_assessment_scores(candidate: dict) -> dict[str, float]:
    """Returns the skill_assessment_scores dict from redrob_signals."""
    return candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})


# ─────────────────────────────────────────────────────────────
# 3. Career summary builder (for reasoning generation)
# ─────────────────────────────────────────────────────────────

def get_career_summary(candidate: dict, max_jobs: int = 3) -> str:
    """
    Short human-readable summary of career for use in reasoning prompts.
    e.g. "ML Engineer at Flipkart (3yr) → Senior MLE at PhonePe (2yr)"
    """
    career = candidate.get("career_history", [])
    parts = []
    for job in career[:max_jobs]:
        months = job.get("duration_months", 0)
        years  = months / 12
        label  = f"{job.get('title', '?')} at {job.get('company', '?')} ({years:.1f}yr)"
        parts.append(label)
    return " → ".join(parts)


def is_consulting_only(candidate: dict) -> bool:
    """
    Returns True if the candidate's entire career is at consulting firms.
    Partially consulting (has at least one product company) returns False.
    """
    from src.config import CONSULTING_FIRMS

    career = candidate.get("career_history", [])
    if not career:
        return False

    for job in career:
        company_lower = job.get("company", "").lower()
        if not any(firm in company_lower for firm in CONSULTING_FIRMS):
            return False   # Found at least one non-consulting company
    return True


def days_since_active(candidate: dict) -> int:
    """Returns number of days since the candidate was last active on platform."""
    from datetime import date
    from src.config import CURRENT_DATE_STR

    today        = date.fromisoformat(CURRENT_DATE_STR)
    last_active  = candidate.get("redrob_signals", {}).get("last_active_date", "2020-01-01")
    try:
        last_dt = date.fromisoformat(last_active)
    except ValueError:
        return 999
    return max(0, (today - last_dt).days)
