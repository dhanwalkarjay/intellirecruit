"""
scorer.py — All scoring components and final weighted score computation.

Components:
  semantic_score   (35%) — embedding cosine similarity with JD ideal text
  skill_score      (30%) — fuzzy match of candidate skills vs required skills
  experience_score (15%) — years of experience + product company flag
  behavioral_score (15%) — platform availability and engagement signals
  location_score   ( 5%) — geographic proximity to Pune/Noida

Penalty multipliers applied after weighted sum:
  honeypot      → ×0.0  (eliminates candidate completely)
  consulting-only → ×0.5
  inactive + not open → ×0.7
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process

from src.config import (
    WEIGHTS,
    REQUIRED_SKILLS_WEIGHTED,
    CONSULTING_FIRMS,
    TARGET_CITIES,
    ACCEPTABLE_CITIES,
    EXP_TARGET_MIN,
    EXP_TARGET_MAX,
    NOTICE_PERIOD_PREFERRED_DAYS,
    INACTIVE_THRESHOLD_DAYS,
    FUZZY_MATCH_THRESHOLD,
    CURRENT_DATE_STR,
)
from src.features import is_consulting_only, days_since_active


# ─────────────────────────────────────────────
# Component 1: Semantic score
# ─────────────────────────────────────────────

def compute_semantic_score(
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
) -> float:
    """
    Cosine similarity between JD ideal-text embedding and candidate embedding.
    Both vectors must be L2-normalised → dot product = cosine sim ∈ [0, 1].
    """
    raw = float(np.dot(jd_embedding, candidate_embedding))
    # Clip to [0, 1] — negative similarity means completely unrelated
    return max(0.0, min(1.0, raw))


# ─────────────────────────────────────────────
# Component 2: Skill score
# ─────────────────────────────────────────────

_PROFICIENCY_WEIGHT = {
    "expert":       1.00,
    "advanced":     0.85,
    "intermediate": 0.65,
    "beginner":     0.35,
}


def compute_skill_score(candidate: dict) -> float:
    """
    Compares candidate skills against REQUIRED_SKILLS_WEIGHTED using fuzzy matching.

    For each required skill:
      - Find best fuzzy match in candidate's skill list (token_sort_ratio ≥ threshold)
      - Weight by proficiency level
      - Boost slightly if candidate has a platform assessment score for that skill
      - Weight by skill importance (2.0 / 1.5 / 1.0)

    Returns normalised score ∈ [0, 1].
    """
    skills         = candidate.get("skills", [])
    assessments    = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    if not skills:
        return 0.0

    # Build lookup: lowercase name → (proficiency_weight, assessment_score_0_to_1)
    skill_lookup: dict[str, tuple[float, float]] = {}
    for s in skills:
        name_lower = s.get("name", "").lower().strip()
        if not name_lower:
            continue
        prof_w     = _PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.5)
        assess_raw = assessments.get(s.get("name", ""), None)
        assess_w   = (assess_raw / 100.0) if assess_raw is not None else None
        skill_lookup[name_lower] = (prof_w, assess_w)

    candidate_skill_names = list(skill_lookup.keys())

    accumulated = 0.0
    max_possible = sum(REQUIRED_SKILLS_WEIGHTED.values())

    for req_skill, req_weight in REQUIRED_SKILLS_WEIGHTED.items():
        req_lower = req_skill.lower()

        # Fuzzy match: find best candidate skill
        result = process.extractOne(
            req_lower,
            candidate_skill_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )

        if result is None:
            continue  # Skill not found — contributes 0

        matched_name, match_score, _ = result
        prof_w, assess_w = skill_lookup[matched_name]

        # Base contribution = importance × proficiency
        contribution = req_weight * prof_w

        # Small boost if they completed a platform assessment for this skill
        if assess_w is not None:
            # Assessment score provides up to +15% boost on the proficiency component
            contribution += req_weight * assess_w * 0.15

        # Partial credit for weaker fuzzy matches (72–85 range)
        if match_score < 85:
            contribution *= 0.85

        accumulated += contribution

    return min(accumulated / max_possible, 1.0)


# ─────────────────────────────────────────────
# Component 3: Experience score
# ─────────────────────────────────────────────

def compute_experience_score(candidate: dict) -> float:
    """
    Evaluates years of experience fit + product-company vs consulting background.

    JD target: 5–9 years total, ideally 4–5 of those in applied ML at product companies.
    JD disqualifier: entire career at consulting firms.
    """
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    yoe      = float(profile.get("years_of_experience", 0))

    # ── Years-of-experience fit ──────────────────────────────────────────────
    if EXP_TARGET_MIN <= yoe <= EXP_TARGET_MAX:
        yoe_score = 1.0
    elif (EXP_TARGET_MIN - 1) <= yoe < EXP_TARGET_MIN:
        yoe_score = 0.80   # 4 years — "some people hit senior judgment at 4"
    elif EXP_TARGET_MAX < yoe <= EXP_TARGET_MAX + 2:
        yoe_score = 0.80   # 10-11 years — still considered
    elif (EXP_TARGET_MIN - 2) <= yoe < (EXP_TARGET_MIN - 1):
        yoe_score = 0.55   # 3 years
    elif yoe > EXP_TARGET_MAX + 2:
        yoe_score = 0.60   # Very senior — may be over-qualified
    else:
        yoe_score = 0.25   # Too junior

    # ── Industry relevance (has worked in AI/ML/Tech product companies?) ─────
    ai_tech_industries = {
        "artificial intelligence", "machine learning", "software",
        "technology", "saas", "fintech", "data", "cloud", "internet",
        "e-commerce", "analytics", "cybersecurity", "edtech", "healthtech",
    }
    relevant_jobs = sum(
        1 for j in career
        if any(ind in j.get("industry", "").lower() for ind in ai_tech_industries)
    )
    industry_ratio  = relevant_jobs / max(len(career), 1)
    industry_bonus  = industry_ratio * 0.15

    # ── Product company bonus ────────────────────────────────────────────────
    # is_consulting_only() returns True only if EVERY job is at a consulting firm
    product_bonus = 0.0 if is_consulting_only(candidate) else 0.10

    raw = yoe_score * 0.75 + industry_bonus + product_bonus
    return min(raw, 1.0)


# ─────────────────────────────────────────────
# Component 4: Behavioral score
# ─────────────────────────────────────────────

def compute_behavioral_score(candidate: dict) -> float:
    """
    Uses the 23 redrob_signals to measure actual hiring availability and engagement.

    The JD explicitly says:
    "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
    response rate is, for hiring purposes, not actually available."

    Sub-components:
      active_score    (40%) — open_to_work + recency of last login
      recruit_score   (30%) — response rate + response time + interview completion
      notice_score    (20%) — notice period vs JD preference (≤30 days)
      engage_score    (10%) — profile completeness + GitHub activity
    """
    from datetime import date

    signals      = candidate.get("redrob_signals", {})
    today        = date.fromisoformat(CURRENT_DATE_STR)
    days_inactive = days_since_active(candidate)
    open_to_work  = signals.get("open_to_work_flag", False)

    # ── Active score ─────────────────────────────────────────────────────────
    if open_to_work:
        if days_inactive <= 14:
            active_score = 1.00
        elif days_inactive <= 30:
            active_score = 0.90
        elif days_inactive <= 60:
            active_score = 0.75
        elif days_inactive <= 90:
            active_score = 0.55
        else:
            active_score = 0.30
    else:
        # Not marked open-to-work — heavily penalise
        if days_inactive <= 30:
            active_score = 0.40
        else:
            active_score = 0.15

    # ── Recruiter engagement score ───────────────────────────────────────────
    resp_rate    = signals.get("recruiter_response_rate", 0.0)
    resp_hours   = signals.get("avg_response_time_hours", 999.0)
    interview_cr = signals.get("interview_completion_rate", 0.0)

    # Response time: ≤4h = 1.0, ≤24h = 0.7, ≤72h = 0.5, >72h = 0.2
    if resp_hours <= 4:
        resp_time_score = 1.00
    elif resp_hours <= 24:
        resp_time_score = 0.70
    elif resp_hours <= 72:
        resp_time_score = 0.50
    elif resp_hours <= 168:
        resp_time_score = 0.30
    else:
        resp_time_score = 0.10

    recruit_score = (
        resp_rate       * 0.50 +
        resp_time_score * 0.20 +
        interview_cr    * 0.30
    )

    # ── Notice period score ──────────────────────────────────────────────────
    notice = signals.get("notice_period_days", 90)
    if notice <= 0:
        notice_score = 1.00   # Immediately available
    elif notice <= 15:
        notice_score = 1.00
    elif notice <= 30:
        notice_score = 0.90   # JD says "we can buy out up to 30 days"
    elif notice <= 60:
        notice_score = 0.60
    elif notice <= 90:
        notice_score = 0.40
    else:
        notice_score = 0.15   # >90 days is very hard to work with

    # ── Platform engagement score ────────────────────────────────────────────
    completeness   = signals.get("profile_completeness_score", 0) / 100.0
    github_raw     = signals.get("github_activity_score", -1)
    github_score   = max(0.0, github_raw) / 100.0   # -1 (no GitHub) → 0
    linkedin_bonus = 0.10 if signals.get("linkedin_connected", False) else 0.0
    verified_bonus = 0.05 if (signals.get("verified_email") and signals.get("verified_phone")) else 0.0

    engage_score = min(
        completeness * 0.50 + github_score * 0.35 + linkedin_bonus + verified_bonus,
        1.0,
    )

    # ── Weighted combination ─────────────────────────────────────────────────
    return (
        active_score   * 0.40 +
        recruit_score  * 0.30 +
        notice_score   * 0.20 +
        engage_score   * 0.10
    )


# ─────────────────────────────────────────────
# Component 5: Location score
# ─────────────────────────────────────────────

def compute_location_score(candidate: dict) -> float:
    """
    Geographic fit with JD location requirements.

    JD: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR acceptable.
    Outside India: case-by-case, no visa sponsorship.
    """
    profile            = candidate.get("profile", {})
    signals            = candidate.get("redrob_signals", {})
    location           = profile.get("location", "").lower()
    country            = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if any(city in location for city in TARGET_CITIES):
        return 1.0   # Already in a preferred/acceptable city

    if any(city in location for city in ACCEPTABLE_CITIES):
        return 0.75 if willing_to_relocate else 0.55

    if country in ("india", "in"):
        # Somewhere in India — relocation possible
        return 0.50 if willing_to_relocate else 0.30

    # Outside India — JD says no visa sponsorship
    return 0.10


# ─────────────────────────────────────────────
# Penalty multiplier
# ─────────────────────────────────────────────

def compute_penalty_multiplier(candidate: dict) -> float:
    """
    Multiplicative penalties applied after the weighted component sum.

    Penalties stack (multiply together).
    """
    multiplier = 1.0
    signals    = candidate.get("redrob_signals", {})

    # Consulting-only career (entire history at IT services firms)
    if is_consulting_only(candidate):
        multiplier *= 0.50

    # Not open to work AND inactive for >INACTIVE_THRESHOLD_DAYS
    open_to_work  = signals.get("open_to_work_flag", False)
    inactive_days = days_since_active(candidate)
    if not open_to_work and inactive_days > INACTIVE_THRESHOLD_DAYS:
        multiplier *= 0.70

    return multiplier


# ─────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────

def score_candidate(
    candidate: dict,
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    is_honeypot: bool = False,
) -> dict:
    """
    Compute the full score breakdown for one candidate.

    Returns a dict with all component scores, penalty, and final_score.
    Honeypots get final_score = 0.0 and are not scored further.
    """
    cid = candidate.get("candidate_id", "UNKNOWN")

    # ── Honeypot elimination ─────────────────────────────────────────────────
    if is_honeypot:
        return {
            "candidate_id":    cid,
            "final_score":     0.0,
            "semantic_score":  0.0,
            "skill_score":     0.0,
            "experience_score":0.0,
            "behavioral_score":0.0,
            "location_score":  0.0,
            "penalty":         0.0,
            "is_honeypot":     True,
        }

    # ── Compute each component ───────────────────────────────────────────────
    sem  = compute_semantic_score(jd_embedding, candidate_embedding)
    sk   = compute_skill_score(candidate)
    exp  = compute_experience_score(candidate)
    beh  = compute_behavioral_score(candidate)
    loc  = compute_location_score(candidate)
    pen  = compute_penalty_multiplier(candidate)

    # ── Weighted sum + penalty ───────────────────────────────────────────────
    raw_score = (
        WEIGHTS["semantic"]    * sem +
        WEIGHTS["skill"]       * sk  +
        WEIGHTS["experience"]  * exp +
        WEIGHTS["behavioral"]  * beh +
        WEIGHTS["location"]    * loc
    )
    final = round(raw_score * pen, 6)

    return {
        "candidate_id":     cid,
        "final_score":      final,
        "semantic_score":   round(sem, 4),
        "skill_score":      round(sk,  4),
        "experience_score": round(exp, 4),
        "behavioral_score": round(beh, 4),
        "location_score":   round(loc, 4),
        "penalty":          round(pen, 4),
        "is_honeypot":      False,
    }
