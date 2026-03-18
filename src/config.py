"""Central configuration for the RAG pipeline."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCS_ROOT: Path = DATA_DIR / "raw" / "docs"
GOLDEN_DATASET_PATH: Path = DATA_DIR / "eval" / "golden_dataset.json"
CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"

# --- Document Loading ---
SUPPORTED_EXTENSIONS: list[str] = [".md", ".mdx"]
MIN_DOC_LENGTH_CHARS: int = 200

# --- Chunking ---
CHUNK_SIZE_TOKENS: int = 600  # target: 500-800 range
CHUNK_OVERLAP_TOKENS: int = 100  # ~100 token overlap
CHARS_PER_TOKEN: float = 4.0  # rough approximation for char-based splitter
CHUNK_SIZE_CHARS: int = int(CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN)  # 2400
CHUNK_OVERLAP_CHARS: int = int(CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN)  # 400

# --- Vector Store ---
COLLECTION_NAME: str = "langchain_docs"
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536

# --- Retrieval ---
TOP_K: int = 5

# --- Generation ---
GENERATION_MODEL: str = "gemini-2.5-flash"
GENERATION_TEMPERATURE: float = 0.2
MAX_OUTPUT_TOKENS: int = 1024

# --- Prompt Versioning ---
ACTIVE_PROMPT_VERSION: str = "v1"
ACTIVE_PROMPT_PATH: Path = PROMPTS_DIR / "rag" / f"{ACTIVE_PROMPT_VERSION}.yaml"

# --- Phase 2: Hybrid Retrieval ---
BM25_K: int = 20  # candidates from BM25
VECTOR_K: int = 20  # candidates from vector search
HYBRID_WEIGHTS: dict[str, float] = {
    "vector": 0.6,
    "bm25": 0.4,
}

# --- Phase 2: Reranking ---
RERANK_INITIAL_K: int = 20  # total candidates before reranking
RERANK_FINAL_K: int = 5  # results after reranking
