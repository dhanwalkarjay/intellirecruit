
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    JD_IDEAL_TEXT,
    REQUIRED_SKILLS_WEIGHTED,
    TARGET_CITIES,
    EXP_TARGET_MIN,
    EXP_TARGET_MAX,
    EMBEDDING_MODEL,
)
from src.embedder import Embedder


ARTIFACTS_DIR = Path("artifacts")


def save_artifacts(jd_embedding: np.ndarray) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Save embedding
    emb_path = ARTIFACTS_DIR / "jd_embedding.npy"
    np.save(str(emb_path), jd_embedding)
    print(f"  ✓ Saved JD embedding → {emb_path}  (shape: {jd_embedding.shape})")

    # Save structured JD metadata
    jd_meta = {
        "role":             "Senior AI Engineer",
        "company":          "Redrob AI (Series A)",
        "locations":        list(TARGET_CITIES),
        "experience_range": [EXP_TARGET_MIN, EXP_TARGET_MAX],
        "required_skills":  list(REQUIRED_SKILLS_WEIGHTED.keys()),
        "ideal_text":       JD_IDEAL_TEXT,
        "embedding_model":  EMBEDDING_MODEL,
        "embedding_shape":  list(jd_embedding.shape),
    }
    meta_path = ARTIFACTS_DIR / "jd_parsed.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(jd_meta, f, indent=2)
    print(f"  ✓ Saved JD metadata  → {meta_path}")


def verify_embedding(embedder: Embedder, jd_embedding: np.ndarray) -> None:
    """
    Quick sanity check: relevant candidate text should score higher than irrelevant one.
    """
    print("\n── Verification: similarity spot-check ─────────────────────────────")

    test_cases = [
        (
            "strong_match",
            "Senior ML Engineer specialising in production vector search systems. "
            "Built FAISS-based retrieval at scale using sentence-transformers and hybrid "
            "dense-sparse ranking. Shipped NDCG-optimised ranking to 10M users. Pune-based.",
        ),
        (
            "weak_match",
            "Graphic Designer with 8 years experience in Adobe suite. "
            "Expert in Photoshop, Illustrator, and brand identity. "
            "No programming or AI experience.",
        ),
        (
            "partial_match",
            "Data Scientist with 5 years experience in Python and scikit-learn. "
            "Built classification models for e-commerce. Some NLP work. "
            "No explicit retrieval or ranking system experience.",
        ),
    ]

    for label, text in test_cases:
        emb = embedder.embed(text)
        sim = float(np.dot(jd_embedding, emb))
        bar = "█" * int(sim * 40)
        print(f"  {label:<15} sim={sim:.4f}  {bar}")

    print()
    print("  Expected order: strong_match > partial_match > weak_match")
    print("  If order is wrong, review JD_IDEAL_TEXT in src/config.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare JD artifacts for IntelliRecruit.")
    parser.add_argument("--verify", action="store_true", help="Run similarity spot-check after saving.")
    args = parser.parse_args()

    print("=" * 60)
    print("  IntelliRecruit — JD Preparation (Phase 1)")
    print("=" * 60)
    print(f"\n  Model : {EMBEDDING_MODEL}")
    print(f"  Role  : Senior AI Engineer @ Redrob AI")
    print(f"  Text  : {len(JD_IDEAL_TEXT)} characters\n")

    print("[1/2] Loading embedding model...")
    embedder = Embedder()

    print("[2/2] Embedding JD ideal-candidate text...")
    jd_embedding = embedder.embed(JD_IDEAL_TEXT)

    print("\n── Saving artifacts ────────────────────────────────────────────────")
    save_artifacts(jd_embedding)

    if args.verify:
        verify_embedding(embedder, jd_embedding)

    print("\n✅ Phase 1 complete. You can now run rank.py (no internet required).")
    print(f"   Artifacts saved to: {ARTIFACTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
