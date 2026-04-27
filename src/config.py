from pathlib import Path

# Base directory is the project root (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 300
TOP_K = 5
PDF_PATH = BASE_DIR / "data" / "NUST-UG-HANDBOOK.pdf"
CHUNKS_FILE = BASE_DIR / "chunks.json"