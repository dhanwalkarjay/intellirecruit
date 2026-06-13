"""
config.py — Central configuration for IntelliRecruit ranking system.
Edit weights and constants here to tune the ranker without touching logic files.
"""

# ─────────────────────────────────────────────
# Scoring weights (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHTS = {
    "semantic":    0.35,   # Cosine sim: candidate embedding vs JD ideal text
    "skill":       0.30,   # Fuzzy match: candidate skills vs required skills
    "experience":  0.15,   # Years of experience + product company flag
    "behavioral":  0.15,   # Platform availability + engagement signals
    "location":    0.05,   # City / country proximity to Pune/Noida
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ─────────────────────────────────────────────
# Embedding model (must be downloadable via pip, runs locally, no API)
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-d, fast on CPU, good quality

# ─────────────────────────────────────────────
# JD: Ideal candidate text (used for semantic embedding)
# Derived from careful reading of job_description.docx
# ─────────────────────────────────────────────
JD_IDEAL_TEXT = (
    "Senior AI Engineer with 6 to 8 years total experience, of which 4 to 5 years "
    "are in applied machine learning roles at product companies, not consulting firms. "
    "Has shipped at least one production ranking, search, or recommendation system to "
    "real users at meaningful scale. Deep expertise in vector embeddings using "
    "sentence-transformers, BGE, E5, or OpenAI embeddings. Production experience with "
    "vector databases and hybrid retrieval infrastructure such as FAISS, Pinecone, "
    "Weaviate, Milvus, Qdrant, Elasticsearch, or OpenSearch. Hands-on experience "
    "designing evaluation frameworks for ranking systems including NDCG, MRR, MAP, "
    "offline to online correlation, and A/B test interpretation. Strong Python and "
    "software engineering skills. Experience with LLM fine-tuning using LoRA or QLoRA. "
    "Familiar with learning-to-rank models using XGBoost or neural approaches. "
    "Background in NLP and information retrieval, not primarily computer vision or "
    "speech. Open to work and actively seeking new opportunities. Located in or willing "
    "to relocate to Pune, Noida, Hyderabad, Mumbai, or Delhi NCR in India. "
    "Prefers hybrid or flexible work arrangement. Short notice period preferred, "
    "ideally under 30 days. Scrappy product engineering mindset, ships working systems "
    "quickly, iterates on real user feedback. Has open source contributions or external "
    "validation of technical work."
)

# ─────────────────────────────────────────────
# Required skills with weights
# must-have = 2.0, strongly-preferred = 1.5, nice-to-have = 1.0
# ─────────────────────────────────────────────
REQUIRED_SKILLS_WEIGHTED = {
    # Must-have (2.0)
    "embeddings":              2.0,
    "sentence transformers":   2.0,
    "vector database":         2.0,
    "FAISS":                   2.0,
    "retrieval":               2.0,
    "ranking":                 2.0,
    "NDCG":                    2.0,
    "Python":                  2.0,
    "information retrieval":   2.0,
    "hybrid search":           2.0,
    "semantic search":         2.0,
    # Strongly preferred (1.5)
    "NLP":                     1.5,
    "recommendation system":   1.5,
    "search":                  1.5,
    "machine learning":        1.5,
    "deep learning":           1.5,
    "Elasticsearch":           1.5,
    # Nice-to-have (1.0)
    "LLM":                     1.0,
    "fine-tuning":             1.0,
    "LoRA":                    1.0,
    "learning to rank":        1.0,
    "A/B testing":             1.0,
    "Pinecone":                1.0,
    "Milvus":                  1.0,
    "Weaviate":                1.0,
    "Qdrant":                  1.0,
    "OpenSearch":              1.0,
    "transformers":            1.0,
    "PyTorch":                 1.0,
    "distributed systems":     1.0,
    "inference optimization":  1.0,
    "open source":             1.0,
}

# ─────────────────────────────────────────────
# Consulting firms — career entirely here = 0.5× penalty
# (JD explicitly says: "consulting firms — bad fit in both directions")
# ─────────────────────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "birlasoft",
    "mindtree",   # borderline — keep lower weight
    "niit technologies",
}

# ─────────────────────────────────────────────
# Location preferences
# ─────────────────────────────────────────────
TARGET_CITIES = {
    "pune", "noida", "delhi", "new delhi", "hyderabad", "mumbai",
    "gurugram", "gurgaon", "ncr", "delhi ncr",
}
ACCEPTABLE_CITIES = {
    "bangalore", "bengaluru", "chennai", "kolkata",
    "ahmedabad", "jaipur", "indore",
}

# ─────────────────────────────────────────────
# Experience year range (from JD)
# ─────────────────────────────────────────────
EXP_TARGET_MIN = 5
EXP_TARGET_MAX = 9

# ─────────────────────────────────────────────
# Behavioral signal thresholds
# ─────────────────────────────────────────────
CURRENT_DATE_STR = "2026-06-11"   # Hardcoded — no datetime.today() during offline ranking

NOTICE_PERIOD_PREFERRED_DAYS = 30   # JD: "we'd love sub-30-day notice"
INACTIVE_THRESHOLD_DAYS      = 90   # Beyond this → down-weight if not open_to_work

# ─────────────────────────────────────────────
# Batch size for embedding (tune based on RAM)
# 512 is safe for 16 GB RAM with MiniLM-L6-v2
# ─────────────────────────────────────────────
EMBED_BATCH_SIZE = 512

# ─────────────────────────────────────────────
# Fuzzy match threshold — below this, skill is not counted
# ─────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD = 72   # 0–100 scale
