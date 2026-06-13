"""
honeypot.py — Detect candidates with subtly impossible profiles.

The dataset contains ~80 honeypot candidates that the ground truth forces to
relevance tier 0.  If >10% of your top-100 are honeypots → instant disqualification.

A good ranker naturally avoids them; this module makes it explicit.
Honeypots get final_score = 0.0 regardless of other signals.
"""

from __future__ import annotations
from datetime import date


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Returns (is_honeypot: bool, reason: str).

    Rules are based on logical impossibilities in the profile data:
      1. Duration at a single company exceeds stated total experience
      2. Multiple "expert" skills with 0 months of usage
      3. Too many expert skills simultaneously (implausible breadth)
      4. Total career months heavily exceeds stated years_of_experience
      5. Two jobs with identical start+end dates (duplicate entries)
      6. Career started before candidate could plausibly have finished school
    """
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    yoe      = profile.get("years_of_experience", 0)

    # ── Rule 1: Single job duration > total stated experience ────────────────
    for job in career:
        job_months = job.get("duration_months", 0)
        # Allow 12-month grace (e.g. fractional years rounding)
        if job_months > (yoe * 12) + 12:
            return True, (
                f"Job at '{job.get('company', '?')}' claims {job_months} months "
                f"but total experience is only {yoe} years"
            )

    # ── Rule 2: Expert skill with 0 months used — 3+ such skills is a red flag
    expert_zero = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(expert_zero) >= 3:
        return True, (
            f"Claims expert proficiency in {len(expert_zero)} skills "
            f"with 0 months of usage: {expert_zero[:3]}"
        )

    # ── Rule 3: Implausible breadth — expert in 10+ skills ──────────────────
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        return True, (
            f"Claims expert proficiency in {len(expert_skills)} skills simultaneously "
            f"(max plausible is ~5-6)"
        )

    # ── Rule 4: Sum of all career durations >> stated YoE ───────────────────
    # Allow up to 2 years of overlap (parallel projects, consulting stints)
    total_months = sum(j.get("duration_months", 0) for j in career)
    max_allowed  = (yoe + 3) * 12
    if total_months > max_allowed and len(career) > 1:
        return True, (
            f"Sum of career durations ({total_months} months) exceeds "
            f"stated experience ({yoe} years) by more than 3 years"
        )

    # ── Rule 5: Duplicate job entries (same start + end date) ───────────────
    date_pairs: list[tuple[str, str | None]] = [
        (j.get("start_date", ""), j.get("end_date", "")) for j in career
    ]
    seen: set[tuple[str, str | None]] = set()
    for pair in date_pairs:
        if pair in seen and pair[0]:  # ignore (None, None) blanks
            return True, f"Two jobs share identical start+end dates: {pair}"
        seen.add(pair)

    # ── Rule 6: Job start date precedes plausible graduation ────────────────
    # Minimum graduation age ~21. If a job started when candidate was < 16 → flag.
    # We estimate birth year from: current year - yoe - 22 (median graduation age)
    current_year   = 2026
    est_birth_year = current_year - int(yoe) - 22
    for job in career:
        start = job.get("start_date", "")
        if start:
            try:
                start_year = int(start[:4])
                if start_year < est_birth_year + 16:
                    return True, (
                        f"Job at '{job.get('company', '?')}' started in {start_year}, "
                        f"but candidate (est. born ~{est_birth_year}) would have been "
                        f"under 16"
                    )
            except (ValueError, TypeError):
                pass

    return False, ""


def honeypot_score_override(is_honeypot: bool) -> float | None:
    """
    Returns 0.0 if honeypot (to be used as final_score directly).
    Returns None if not a honeypot (score computed normally).
    """
    return 0.0 if is_honeypot else None
