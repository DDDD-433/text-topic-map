"""Central configuration for the text-topic-map pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"

CLEAN_PARQUET = DATA_PROCESSED_DIR / "clean.parquet"
EMBEDDED_TOPICS_PARQUET = DATA_PROCESSED_DIR / "embedded_topics.parquet"
TOPIC_SUMMARY_CSV = DATA_PROCESSED_DIR / "topic_summary.csv"
DATA_SUMMARY_TXT = ASSETS_DIR / "data_summary.txt"

DATASET_NAME = "SetFit/bbc-news"
MAX_ROWS = 5000

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 15
RANDOM_STATE = 42
