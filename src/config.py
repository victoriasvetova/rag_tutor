import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ====== LLM / Embeddings ======
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # ====== Paths ======
    DB_PATH = os.getenv("DB_PATH")
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")

    # ====== Chunking (ВАЖНО ДЛЯ ОБУЧЕНИЯ) ======
    # Меньшие чанки → лучше объяснения
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

    # ====== Retrieval (Teaching RAG) ======
    # Берём больше документов → LLM сама отфильтрует
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 12))
    TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", 8))

    # ====== RAG MODE ======
    # search  — строгий поиск
    # tutor   — обучающий режим (по ТЗ)
    RAG_MODE = os.getenv("RAG_MODE", "tutor")


settings = Config()