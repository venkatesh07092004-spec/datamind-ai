"""Centralized configuration and resource limits for DataMind AI."""

MAX_FILE_SIZE_MB: float = 50.0
MAX_ROWS: int = 500_000
MAX_COLUMNS: int = 300
MAX_QUERY_ROWS: int = 10_000

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: str = "llama3.2"
OLLAMA_TIMEOUT_SECONDS: int = 20
