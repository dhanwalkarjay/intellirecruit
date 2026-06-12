

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from src.features import get_career_summary, days_since_active


# ─────────────────────────────────────────────
# Load top-100 candidate IDs from submission CSV
# ─────────────────────────────────────────────

def load_submission(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ─────────────────────────────────────────────
# Build a candidate lookup dict from JSONL
# ─────────────────────────────────────────────

def build_candidate_lookup(jsonl_path: str, target_ids: set[str]) -> dict[str, dict]:
    """Stream JSONL and collect only the candidates we need."""
    lookup = {}
    opener = gzip.open if jsonl_path.endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = c.get("candidate_id", "")
            if cid in target_ids:
                lookup[cid] = c
            if len(lookup) == len(target_ids):
                break  # Found all — stop early
    return lookup


# ─────────────────────────────────────────────
# Build prompt for a single candidate
# ─────────────────────────────────────────────

def build_reasoning_prompt(candidate: dict, rank: int, score: float) -> str:
    p        = candidate.get("profile", {})
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})
    career   = candidate.get("career_history", [])

    # Build concise skill list (top 8 by proficiency then endorsements)
    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    sorted_skills = sorted(
        skills,
        key=lambda s: (prof_order.get(s.get("proficiency", ""), 0), s.get("endorsements", 0)),
        reverse=True,
    )
    top_skill_names = ", ".join(s["name"] for s in sorted_skills[:8])

    career_summary = get_career_summary(candidate, max_jobs=3)
    inactive_days  = days_since_active(candidate)
    notice         = signals.get("notice_period_days", "?")
    open_to_work   = signals.get("open_to_work_flag", False)
    location       = p.get("location", "?")
    yoe            = p.get("years_of_experience", "?")
    current_title  = p.get("current_title", "?")
    current_co     = p.get("current_company", "?")

    # Tone guidance based on rank
    if rank <= 10:
        tone = "enthusiastic and specific — this is a strong match"
    elif rank <= 30:
        tone = "positive but measured — good fit with minor gaps"
    elif rank <= 60:
        tone = "neutral and factual — moderate fit"
    else:
        tone = "honest and matter-of-fact — borderline fit, acknowledge gaps"

    prompt = f"""You are writing a 1-2 sentence recruiter reasoning note for a candidate ranked #{rank} out of 100.

ROLE: Senior AI Engineer at Redrob AI (Series A), Pune/Noida
KEY REQUIREMENTS: Production embeddings/retrieval systems, vector DBs (FAISS/Pinecone/Weaviate), 
Python, evaluation frameworks (NDCG/MAP/MRR), 5-9 years exp, product companies (not consulting), 
open to work, short notice period preferred.

CANDIDATE PROFILE:
- Current: {current_title} at {current_co}
- Experience: {yoe} years
- Location: {location}
- Top skills: {top_skill_names}
- Career path: {career_summary}
- Open to work: {open_to_work} | Last active: {inactive_days} days ago | Notice: {notice} days

RANK: #{rank} | SCORE: {score:.4f}
TONE: {tone}

Rules:
1. Reference SPECIFIC facts — name actual skills, years, company type
2. Connect directly to JD requirements
3. For ranks 50+: honestly mention the main gap (missing retrieval experience, consulting-only career, etc.)
4. Do NOT invent facts not in the profile above
5. Keep to 1-2 sentences, max 40 words total
6. Do not start with "Candidate" or use their name (anonymized)
7. Vary phrasing — do not use a template

Write ONLY the reasoning sentence(s), nothing else."""
    return prompt


# ─────────────────────────────────────────────
# Groq API call (with retry + fallback)
# ─────────────────────────────────────────────

def call_groq(prompt: str, client) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=80,
            )
            text = resp.choices[0].message.content.strip()
            # Clean up: remove quotes if LLM wrapped in them
            text = text.strip('"').strip("'")
            return text
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"      Groq error (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                return f"{candidate_fallback_reasoning()}"
    return ""


def candidate_fallback_reasoning() -> str:
    return "Profile reviewed; score reflects best available signal match against JD requirements."


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning for top-100 candidates.")
    parser.add_argument("--submission",  default="output/submission.csv",       help="Input CSV from rank.py")
    parser.add_argument("--candidates",  default="candidates.jsonl",            help="Candidates JSONL/gz")
    parser.add_argument("--out",         default="output/submission_final.csv", help="Output CSV with reasoning")
    parser.add_argument("--dry-run",     action="store_true",                   help="Print prompts without calling API")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key and not args.dry_run:
        print("❌ GROQ_API_KEY not set in .env file.")
        print("   Run: cp .env.example .env  then add your key.")
        sys.exit(1)

    print("=" * 60)
    print("  IntelliRecruit — Reasoning Generation (Phase 3)")
    print("=" * 60)

    # Load submission
    print(f"\n[1/4] Loading submission: {args.submission}")
    submission_rows = load_submission(args.submission)
    print(f"      Found {len(submission_rows)} rows")

    # Get target IDs
    target_ids = {row["candidate_id"] for row in submission_rows}

    # Build candidate lookup
    print(f"[2/4] Loading candidate profiles from: {args.candidates}")
    lookup = build_candidate_lookup(args.candidates, target_ids)
    print(f"      Loaded {len(lookup)} candidate profiles")

    missing = target_ids - set(lookup.keys())
    if missing:
        print(f"      ⚠️  {len(missing)} candidate IDs not found in JSONL: {list(missing)[:5]}")

    # Init Groq client
    if not args.dry_run:
        from groq import Groq
        client = Groq(api_key=api_key)
    else:
        client = None

    # Generate reasoning for each row
    print(f"[3/4] Generating reasoning (Groq Llama 3.1 70B)...")
    print(f"      {'DRY RUN — no API calls' if args.dry_run else 'Live API calls'}")
    print()

    updated_rows = []
    for i, row in enumerate(submission_rows):
        cid   = row["candidate_id"]
        rank  = int(row["rank"])
        score = float(row["score"])
        candidate = lookup.get(cid)

        if candidate is None:
            reasoning = "Profile not found in candidate pool."
        else:
            prompt = build_reasoning_prompt(candidate, rank, score)

            if args.dry_run:
                print(f"  Rank {rank:>3} ({cid}):")
                print(f"    {prompt[:200]}...\n")
                reasoning = f"[DRY RUN] Rank {rank} reasoning placeholder."
            else:
                reasoning = call_groq(prompt, client)
                # Rate limit: Groq free tier is generous but add small delay
                time.sleep(0.3)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/100] {cid} rank={rank} — {reasoning[:70]}...")

        updated_rows.append({
            "candidate_id": cid,
            "rank":         row["rank"],
            "score":        row["score"],
            "reasoning":    reasoning,
        })

    # Write final CSV
    print(f"\n[4/4] Writing final submission to: {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\n✅ Done! Final submission saved to: {Path(args.out).resolve()}")
    print(f"\n   Next step: validate before submitting:")
    print(f"   python validate_submission.py {args.out}\n")


if __name__ == "__main__":
    main()
