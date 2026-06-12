"""
rank.py — Phase 2: Main ranking script (OFFLINE — no network, CPU only, ≤5 min).

Reads candidates.jsonl (or .jsonl.gz), scores all 100K candidates,
and writes the top-100 ranked CSV to the output path.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./output/submission.csv
  python rank.py --candidates ./candidates.jsonl.gz --out ./output/submission.csv

Compute constraints (from submission_spec):
  • No external API calls
  • CPU only (no GPU required)
  • ≤16 GB RAM
  • Must complete in ≤5 minutes

Design:
  • Stream JSONL line by line (never load full 465 MB into RAM)
  • Batch candidates for efficient sentence-transformer encoding (batch=512)
  • Scoring is pure numpy — no LLM calls
  • Reasoning column left empty here; filled by generate_reasoning.py
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import EMBED_BATCH_SIZE
from src.embedder import Embedder
from src.features import build_candidate_text
from src.honeypot import detect_honeypot
from src.scorer import score_candidate


# ─────────────────────────────────────────────
# JSONL streaming (handles both plain and gzip)
# ─────────────────────────────────────────────

def stream_candidates(path: str):
    """Yield one candidate dict per line from .jsonl or .jsonl.gz file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines silently


# ─────────────────────────────────────────────
# Batch processor
# ─────────────────────────────────────────────

def process_batch(
    batch_candidates: list[dict],
    batch_texts: list[str],
    jd_embedding: np.ndarray,
    embedder: Embedder,
    honeypot_map: dict[str, tuple[bool, str]],
) -> list[dict]:
    """
    Embed a batch of candidate texts and score each candidate.
    Returns list of score dicts.
    """
    embeddings = embedder.embed_batch(batch_texts, batch_size=EMBED_BATCH_SIZE)
    results = []

    for candidate, emb in zip(batch_candidates, embeddings):
        cid         = candidate.get("candidate_id", "")
        is_hp, _    = honeypot_map.get(cid, (False, ""))
        score_dict  = score_candidate(candidate, jd_embedding, emb, is_honeypot=is_hp)
        results.append(score_dict)

    return results


# ─────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────

def write_submission_csv(top_100: list[dict], out_path: str) -> None:
    """
    Write the top-100 ranked candidates to CSV.

    Format required by validate_submission.py:
      candidate_id, rank, score, reasoning
      - Exactly 100 data rows
      - Ranks 1–100, each exactly once
      - Scores non-increasing
      - reasoning column: filled later by generate_reasoning.py
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure scores are strictly non-increasing (may be equal, but never go up)
    # Tie-break: candidate_id ascending (required by validator)
    top_100.sort(key=lambda x: (-x["final_score"], x["candidate_id"].upper()))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(top_100, start=1):
            writer.writerow([row["candidate_id"], rank, f"{row['final_score']:.6f}", ""]) 


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IntelliRecruit — Rank candidates against JD.")
    parser.add_argument("--candidates", default="candidates.jsonl",  help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out",        default="output/submission.csv", help="Output CSV path")
    parser.add_argument("--top-k",      type=int, default=100, help="Number of top candidates to output (default 100)")
    parser.add_argument("--debug",      action="store_true", help="Print score breakdown for top-20")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print("  IntelliRecruit — Candidate Ranking (Phase 2)")
    print("=" * 60)

    # ── Load JD embedding ────────────────────────────────────────────────────
    emb_path = Path("artifacts/jd_embedding.npy")
    if not emb_path.exists():
        print("\n❌ ERROR: artifacts/jd_embedding.npy not found.")
        print("   Run `python prepare_jd.py` first to generate it.\n")
        sys.exit(1)

    print("\n[1/4] Loading JD embedding...")
    jd_embedding = np.load(str(emb_path)).astype(np.float32)
    print(f"      Shape: {jd_embedding.shape}")

    # ── Load embedding model ─────────────────────────────────────────────────
    print("[2/4] Loading sentence-transformer model (local, no internet)...")
    embedder = Embedder()
    # Warm up to avoid cold-start timing issues
    _ = embedder.embed("warmup text")
    print(f"      Model ready: {embedder.model_name}")

    # ── Stream, score, collect ───────────────────────────────────────────────
    print(f"[3/4] Streaming and scoring candidates from: {args.candidates}")
    print(f"      Batch size: {EMBED_BATCH_SIZE}")

    all_scores: list[dict]           = []
    batch_candidates: list[dict]     = []
    batch_texts: list[str]           = []
    honeypot_map: dict[str, tuple]   = {}   # cid → (is_hp, reason)

    n_processed    = 0
    n_honeypots    = 0
    LOG_EVERY      = 10_000

    for candidate in stream_candidates(args.candidates):
        cid = candidate.get("candidate_id", "")

        # Honeypot check is fast (no embedding needed)
        is_hp, hp_reason = detect_honeypot(candidate)
        honeypot_map[cid] = (is_hp, hp_reason)
        if is_hp:
            n_honeypots += 1
            # Still need a score entry (final_score=0.0) for accounting
            all_scores.append({
                "candidate_id":     cid,
                "final_score":      0.0,
                "semantic_score":   0.0,
                "skill_score":      0.0,
                "experience_score": 0.0,
                "behavioral_score": 0.0,
                "location_score":   0.0,
                "penalty":          0.0,
                "is_honeypot":      True,
            })
            n_processed += 1
            continue

        # Build embedding text and queue in batch
        text = build_candidate_text(candidate)
        batch_candidates.append(candidate)
        batch_texts.append(text)

        # Process batch when full
        if len(batch_texts) >= EMBED_BATCH_SIZE:
            results = process_batch(batch_candidates, batch_texts, jd_embedding, embedder, honeypot_map)
            all_scores.extend(results)
            batch_candidates, batch_texts = [], []

        n_processed += 1
        if n_processed % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate    = n_processed / elapsed
            eta     = (100_000 - n_processed) / rate if rate > 0 else 0
            print(f"      {n_processed:>7,} processed | {elapsed:5.1f}s elapsed | "
                  f"ETA {eta:4.0f}s | honeypots: {n_honeypots}")

    # Process remaining partial batch
    if batch_texts:
        results = process_batch(batch_candidates, batch_texts, jd_embedding, embedder, honeypot_map)
        all_scores.extend(results)

    elapsed_scoring = time.time() - t_start
    print(f"\n      Total candidates processed : {n_processed:,}")
    print(f"      Honeypots detected          : {n_honeypots}")
    print(f"      Scoring time                : {elapsed_scoring:.1f}s")

    # ── Sort and select top-K ────────────────────────────────────────────────
    print(f"\n[4/4] Ranking and writing top-{args.top_k} to: {args.out}")

    all_scores.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top_k = all_scores[:args.top_k]

    # ── Debug: print top-20 breakdown ────────────────────────────────────────
    if args.debug:
        print("\n── Top-20 score breakdown ──────────────────────────────────────────")
        print(f"  {'Rank':<5} {'CandID':<15} {'Final':<7} {'Sem':<6} {'Skill':<6} "
              f"{'Exp':<6} {'Beh':<6} {'Loc':<6} {'Pen':<5}")
        print("  " + "-" * 70)
        for i, row in enumerate(top_k[:20], 1):
            print(f"  {i:<5} {row['candidate_id']:<15} {row['final_score']:<7.4f} "
                  f"{row['semantic_score']:<6.3f} {row['skill_score']:<6.3f} "
                  f"{row['experience_score']:<6.3f} {row['behavioral_score']:<6.3f} "
                  f"{row['location_score']:<6.3f} {row['penalty']:<5.2f}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    write_submission_csv(top_k, args.out)

    total_time = time.time() - t_start
    print(f"\n✅ Done in {total_time:.1f}s")
    print(f"   Top candidate : {top_k[0]['candidate_id']} (score: {top_k[0]['final_score']:.4f})")
    print(f"   Output saved  : {Path(args.out).resolve()}")
    print(f"\n   Next step: run generate_reasoning.py to fill the reasoning column.")
    print(f"   Then run:  python validate_submission.py {args.out}\n")


if __name__ == "__main__":
    main()
