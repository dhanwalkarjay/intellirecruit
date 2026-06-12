"""
app.py — Convenience CLI wrapper for the full IntelliRecruit pipeline.

Runs all three phases in order, or individual phases on demand.

Usage:
  python app.py                          # Full pipeline (prepare → rank → reasoning)
  python app.py --phase prepare          # Phase 1 only: embed JD
  python app.py --phase rank             # Phase 2 only: rank candidates
  python app.py --phase reasoning        # Phase 3 only: generate reasoning
  python app.py --phase rank --debug     # Rank with top-20 score breakdown
  python app.py --candidates data.jsonl.gz  # Custom candidates file
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(label: str, cmd: list[str]) -> bool:
    """Run a subprocess step, print output live, return True if succeeded."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    t0     = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ Step failed (exit code {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"\n⏱  Step completed in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IntelliRecruit — Full pipeline runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["prepare", "rank", "reasoning", "all"],
        default="all",
        help=(
            "Which phase to run:\n"
            "  prepare   — embed JD, save artifacts (Phase 1)\n"
            "  rank      — score all candidates, output CSV (Phase 2)\n"
            "  reasoning — generate LLM reasoning for top-100 (Phase 3)\n"
            "  all       — run all three in order (default)"
        ),
    )
    parser.add_argument("--candidates", default="candidates.jsonl",          help="Path to candidates JSONL/gz")
    parser.add_argument("--out",        default="output/submission.csv",      help="Intermediate submission CSV")
    parser.add_argument("--final-out",  default="output/submission_final.csv",help="Final submission CSV with reasoning")
    parser.add_argument("--top-k",      type=int, default=100,                help="Number of top candidates to output")
    parser.add_argument("--debug",      action="store_true",                  help="Print score breakdown for top-20")
    parser.add_argument("--verify",     action="store_true",                  help="Run JD similarity spot-check in Phase 1")
    parser.add_argument("--dry-run-reasoning", action="store_true",           help="Print reasoning prompts without API calls")
    args = parser.parse_args()

    total_start = time.time()
    python      = sys.executable  # Use same Python as current venv

    print("\n🔍 IntelliRecruit — Pipeline Runner")
    print(f"   Phase      : {args.phase}")
    print(f"   Candidates : {args.candidates}")
    print(f"   Output     : {args.final_out}")

    steps_run  = 0
    steps_ok   = 0

    # ── Phase 1: Prepare JD ──────────────────────────────────────────────────
    if args.phase in ("prepare", "all"):
        steps_run += 1
        cmd = [python, "src/prepare_jd.py"]
        if args.verify:
            cmd.append("--verify")
        ok = run_step("Phase 1 — Prepare JD Artifacts", cmd)
        if not ok and args.phase == "all":
            print("\n❌ Aborting pipeline — Phase 1 failed.")
            sys.exit(1)
        steps_ok += int(ok)

    # ── Phase 2: Rank ────────────────────────────────────────────────────────
    if args.phase in ("rank", "all"):
        steps_run += 1
        cmd = [
            python, "rank.py",
            "--candidates", args.candidates,
            "--out",        args.out,
            "--top-k",      str(args.top_k),
        ]
        if args.debug:
            cmd.append("--debug")
        ok = run_step("Phase 2 — Rank Candidates (offline, no internet)", cmd)
        if not ok and args.phase == "all":
            print("\n❌ Aborting pipeline — Phase 2 failed.")
            sys.exit(1)
        steps_ok += int(ok)

    # ── Phase 3: Reasoning ───────────────────────────────────────────────────
    if args.phase in ("reasoning", "all"):
        steps_run += 1
        cmd = [
            python, "generate_reasoning.py",
            "--submission", args.out,
            "--candidates", args.candidates,
            "--out",        args.final_out,
        ]
        if args.dry_run_reasoning:
            cmd.append("--dry-run")
        ok = run_step("Phase 3 — Generate Reasoning (Groq LLM, requires internet)", cmd)
        steps_ok += int(ok)

    # ── Validate ─────────────────────────────────────────────────────────────
    if args.phase == "all" and steps_ok == steps_run:
        print(f"\n{'='*60}")
        print("  Validating final submission...")
        print(f"{'='*60}\n")
        validate_cmd = [python, "validate_submission.py", args.final_out]
        subprocess.run(validate_cmd)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {steps_ok}/{steps_run} steps succeeded")
    print(f"  Total time      : {total_time:.1f}s")

    if steps_ok == steps_run and args.phase == "all":
        print(f"\n  ✅ Submission ready: {Path(args.final_out).resolve()}")
        print(f"  Next: rename file to <your_participant_id>.csv before uploading.")
    print(f"{'='*60}\n")

    if steps_ok < steps_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
