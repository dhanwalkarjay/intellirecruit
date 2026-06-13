"""
app.py — Streamlit sandbox demo for IntelliRecruit.
Runs on HuggingFace Spaces. Uses sample_candidates.json (50 candidates).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── Make src/ importable from /app/ on HuggingFace Spaces ──────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="IntelliRecruit — AI Candidate Ranker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 IntelliRecruit")
st.sidebar.caption("INDIA RUNS Hackathon 2026 · Track 1")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Rank Candidates", "📊 Score Breakdown", "🏗️ Architecture"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.info(
    "**Compute constraints met:**\n"
    "- ✅ No API calls during ranking\n"
    "- ✅ CPU only\n"
    "- ✅ < 5 min on 100K candidates\n"
    "- ✅ Honeypot detection active"
)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from src.embedder import Embedder
    return Embedder()


@st.cache_data(show_spinner="Loading JD embedding...")
def load_jd_embedding():
    path = ROOT / "artifacts" / "jd_embedding.npy"
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    return None


@st.cache_data(show_spinner="Loading sample candidates...")
def load_sample_candidates():
    for p in [ROOT / "sample_candidates.json", ROOT / "data" / "sample_candidates.json"]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return []


# ── Live ranking on sample data ───────────────────────────────────────────────
def rank_candidates_demo(candidates, jd_text):
    from src.features import build_candidate_text
    from src.honeypot import detect_honeypot
    from src.scorer import score_candidate

    embedder     = load_embedder()
    jd_embedding = embedder.embed(jd_text)
    texts        = [build_candidate_text(c) for c in candidates]

    with st.spinner(f"Embedding {len(candidates)} candidates..."):
        embeddings = embedder.embed_batch(texts, batch_size=64, show_progress=False)

    results = []
    for candidate, emb in zip(candidates, embeddings):
        is_hp, hp_reason = detect_honeypot(candidate)
        score = score_candidate(candidate, jd_embedding, emb, is_honeypot=is_hp)
        score["_candidate"] = candidate
        score["_hp_reason"] = hp_reason
        results.append(score)

    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🔍 IntelliRecruit")
    st.subheader("Intelligent Candidate Discovery & Ranking — INDIA RUNS Hackathon 2026")
    st.markdown("> **Track 1: The Data & AI Challenge** — Moving beyond keyword filters.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates Ranked", "100,000")
    col2.metric("Runtime (100K)", "< 5 min")
    col3.metric("Scoring Components", "5")
    col4.metric("Honeypot Rules", "6")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### ❌ Traditional ATS / Keyword Filter")
        st.markdown("""
        - Exact keyword match only
        - Misses candidates who describe skills differently
        - Ignores behavioral signals entirely
        - No explanation for ranking decisions
        - Fooled by keyword stuffers
        """)

    with col_b:
        st.markdown("#### ✅ IntelliRecruit")
        st.markdown("""
        - **Semantic similarity** — understands meaning, not words
        - **Fuzzy skill matching** — "ML" = "machine learning"
        - **Behavioral signals** — open-to-work, notice period, response rate
        - **Honeypot detection** — 6 impossibility rules
        - **Explainable scores** — full component breakdown per candidate
        """)

    st.divider()
    st.markdown("### Scoring Formula")
    st.code(
        "final_score = (\n"
        "    0.35 × semantic_score     # Cosine sim vs JD ideal embedding\n"
        "  + 0.30 × skill_score        # Fuzzy match, weighted by proficiency\n"
        "  + 0.15 × experience_score   # YoE fit + product company flag\n"
        "  + 0.15 × behavioral_score   # open_to_work, notice, response rate\n"
        "  + 0.05 × location_score     # Pune/Noida/Hyd/Mum/Delhi NCR\n"
        ") × penalty_multiplier        # honeypot=0.0 | consulting=0.5 | inactive=0.7",
        language="python"
    )
    st.info("👈 Go to **🔍 Rank Candidates** to try it live on sample data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RANK CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Rank Candidates":
    st.title("🔍 Rank Candidates")
    st.caption("Live demo on sample_candidates.json (50 candidates). Full 100K runs via rank.py CLI.")

    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown("#### Job Description")
        use_default = st.checkbox("Use default JD (Senior AI Engineer)", value=True)

        from src.config import JD_IDEAL_TEXT
        if use_default:
            jd_text = JD_IDEAL_TEXT
            st.text_area("JD (read-only)", value=jd_text[:500] + "...", height=180, disabled=True)
        else:
            jd_text = st.text_area(
                "Paste custom JD",
                height=220,
                placeholder="We are looking for a Senior AI Engineer with expertise in "
                            "production embeddings, FAISS, retrieval systems...",
            )

        top_k   = st.slider("Show top N candidates", 5, 50, 10)
        run_btn = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

    with col_right:
        if run_btn and jd_text:
            candidates = load_sample_candidates()
            if not candidates:
                st.error("❌ sample_candidates.json not found in Space files.")
            else:
                results = rank_candidates_demo(candidates, jd_text)
                st.markdown(f"#### Top {top_k} Results")

                for r in results[:top_k]:
                    c    = r["_candidate"]
                    p    = c.get("profile", {})
                    sig  = c.get("redrob_signals", {})
                    is_hp = r["is_honeypot"]

                    with st.expander(
                        f"{'🚫 HONEYPOT  ' if is_hp else ''}#{r['rank']}  —  "
                        f"{p.get('current_title','?')} @ {p.get('current_company','?')}  "
                        f"·  Score: **{r['final_score']:.3f}**",
                        expanded=(r["rank"] <= 3),
                    ):
                        if is_hp:
                            st.error(f"🚫 Honeypot: {r['_hp_reason']}")
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Overall",   f"{r['final_score']:.3f}")
                            c2.metric("Semantic",  f"{r['semantic_score']:.3f}")
                            c3.metric("Skill",     f"{r['skill_score']:.3f}")
                            c4, c5, c6 = st.columns(3)
                            c4.metric("Experience",f"{r['experience_score']:.3f}")
                            c5.metric("Behavioral",f"{r['behavioral_score']:.3f}")
                            c6.metric("Location",  f"{r['location_score']:.3f}")

                            st.markdown(
                                f"**{p.get('years_of_experience','?')} yrs** · "
                                f"📍 {p.get('location','?')} · "
                                f"{'✅ Open to work' if sig.get('open_to_work_flag') else '⏸️ Not open'} · "
                                f"⏱️ {sig.get('notice_period_days','?')}d notice · "
                                f"Last active: {sig.get('last_active_date','?')}"
                            )
                            top_skills = sorted(
                                c.get("skills", []),
                                key=lambda s: s.get("endorsements", 0), reverse=True
                            )[:6]
                            st.write("**Skills:** " + "  |  ".join(
                                f"`{s['name']} ({s.get('proficiency','')})`" for s in top_skills
                            ))
                            if r["penalty"] < 1.0:
                                st.warning(f"⚠️ Penalty applied: ×{r['penalty']:.2f}")
        else:
            st.info("👈 Click **🚀 Rank Candidates** to see results")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SCORE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Score Breakdown":
    st.title("📊 Score Breakdown")
    st.caption("Explore the 5-component score for any candidate in the sample set.")

    candidates = load_sample_candidates()
    if not candidates:
        st.error("❌ sample_candidates.json not found.")
    else:
        from src.config import JD_IDEAL_TEXT, WEIGHTS
        from src.features import build_candidate_text
        from src.honeypot import detect_honeypot
        from src.scorer import (
            compute_semantic_score, compute_skill_score,
            compute_experience_score, compute_behavioral_score,
            compute_location_score, compute_penalty_multiplier,
        )

        embedder     = load_embedder()
        jd_embedding = embedder.embed(JD_IDEAL_TEXT)

        labels = [
            f"{c['candidate_id']} — {c['profile'].get('current_title','?')} @ {c['profile'].get('current_company','?')}"
            for c in candidates[:20]
        ]
        idx       = st.selectbox("Select a candidate", range(len(labels)), format_func=lambda i: labels[i])
        candidate = candidates[idx]
        is_hp, hp_reason = detect_honeypot(candidate)
        emb = embedder.embed(build_candidate_text(candidate))

        sem  = compute_semantic_score(jd_embedding, emb)
        sk   = compute_skill_score(candidate)
        exp  = compute_experience_score(candidate)
        beh  = compute_behavioral_score(candidate)
        loc  = compute_location_score(candidate)
        pen  = compute_penalty_multiplier(candidate)
        raw  = WEIGHTS["semantic"]*sem + WEIGHTS["skill"]*sk + WEIGHTS["experience"]*exp + WEIGHTS["behavioral"]*beh + WEIGHTS["location"]*loc
        final = raw * pen

        p   = candidate.get("profile", {})
        sig = candidate.get("redrob_signals", {})

        st.subheader(f"{p.get('current_title','?')} @ {p.get('current_company','?')}")
        if is_hp:
            st.error(f"🚫 HONEYPOT: {hp_reason}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Score", f"{final:.4f}")
            st.markdown(f"""
            - **Experience:** {p.get('years_of_experience','?')} years
            - **Location:** {p.get('location','?')}
            - **Open to work:** {'✅' if sig.get('open_to_work_flag') else '❌'}
            - **Notice period:** {sig.get('notice_period_days','?')} days
            - **Last active:** {sig.get('last_active_date','?')}
            - **Response rate:** {sig.get('recruiter_response_rate','?')}
            """)

        with col2:
            components = {
                "Semantic (35%)":   sem,
                "Skill (30%)":      sk,
                "Experience (15%)": exp,
                "Behavioral (15%)": beh,
                "Location (5%)":    loc,
            }
            for name, val in components.items():
                st.progress(val, text=f"{name}: {val:.3f}")
            if pen < 1.0:
                st.warning(f"Penalty multiplier: ×{pen:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.title("🏗️ System Architecture")

    st.markdown("### Two-Phase Design (meets all compute constraints)")
    st.code("""
PHASE 1 — Offline pre-computation (run once, internet OK)
  prepare_jd.py → embeds JD ideal text → artifacts/jd_embedding.npy

PHASE 2 — rank.py (offline, no internet, CPU only, ≤5 min)
  candidates.jsonl (100K)
    → stream line-by-line (memory efficient)
    → detect_honeypot() [6 rules]
    → build_candidate_text() [rich embedding paragraph]
    → embed_batch() [MiniLM-L6-v2, batch=512]
    → score_candidate() [5 components + penalties]
    → sort all 100K → top-100 → submission.csv

PHASE 3 — generate_reasoning.py (offline, post-ranking)
  top-100 → Groq Llama 3.3 70B → factual reasoning → final CSV
    """, language="text")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Scoring Weights")
        st.markdown("""
        | Component | Weight | Signal |
        |---|---|---|
        | Semantic | **35%** | MiniLM cosine sim vs JD text |
        | Skill match | **30%** | Fuzzy + proficiency + assessments |
        | Experience | **15%** | YoE fit + product vs consulting |
        | Behavioral | **15%** | open_to_work, notice, response rate |
        | Location | **5%** | Pune/Noida/Hyd/Mum/Delhi NCR |
        """)
    with col2:
        st.markdown("### Penalty Multipliers")
        st.markdown("""
        | Condition | Multiplier |
        |---|---|
        | Honeypot detected | **×0.0** |
        | Consulting-only career | **×0.5** |
        | Inactive + not open | **×0.7** |
        """)

    st.divider()
    st.markdown("### Honeypot Detection (6 Rules)")
    st.markdown("""
    1. Single job duration > total stated YoE  
    2. ≥3 skills claimed expert with 0 months usage  
    3. Expert in 10+ skills simultaneously  
    4. Total career months >> stated YoE by 3+ years  
    5. Two jobs share identical start+end dates  
    6. Job started when candidate would be under 16  
    """)

    st.divider()
    st.markdown("### Tech Stack")
    st.markdown("""
    | Tool | Purpose |
    |---|---|
    | `all-MiniLM-L6-v2` | Local embeddings, no API, fast on CPU |
    | `rapidfuzz` | Fuzzy skill matching |
    | `numpy` | Cosine similarity via dot product |
    | `Groq Llama 3.3 70B` | Reasoning generation (post-ranking only) |
    | `Streamlit` | This demo |
    | HuggingFace Spaces | Free hosting |
    """)