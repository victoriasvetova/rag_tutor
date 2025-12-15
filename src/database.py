import shutil
import os
from typing import List, Dict, Tuple
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger
from src.config import settings

class VectorDatabase:
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )
        self.db = None

    def _get_db(self):
        """Ленивая инициализация: подключаемся только когда нужно"""
        if self.db is None:
            self.db = Chroma(
                persist_directory=settings.DB_PATH,
                embedding_function=self.embedding_model
            )
        return self.db

    def rebuild(self, data: List[Dict[str, str]]):
        """Полная перестройка базы данных"""
        self.db = None # Сбрасываем подключение
        
        logger.warning("🗑 Удаляю старую базу данных...")
        if os.path.exists(settings.DB_PATH):
            try:
                shutil.rmtree(settings.DB_PATH)
            except Exception as e:
                logger.error(f"Не удалось удалить папку базы: {e}")
                return

        logger.info("🔪 Нарезаю текст на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        documents = []
        for item in data:
            chunks = text_splitter.create_documents(
                texts=[item['content']],
                metadatas=[{"source": item['source'], "title": item['title']}]
            )
            documents.extend(chunks)

        logger.info(f"🧩 Создано {len(documents)} чанков. Начинаю векторизацию...")
        
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=settings.DB_PATH
        )
        logger.success("💾 База данных успешно создана.")

    def search(self, query: str, k: int) -> List[Document]:
        """Обычный поиск"""
        return self._get_db().similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Поиск с возвратом оценки уверенности (distance)"""
        return self._get_db().similarity_search_with_score(query, k=k)