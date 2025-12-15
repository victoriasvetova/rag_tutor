from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from src.database import VectorDatabase
from src.config import settings
from loguru import logger


class RAGEngine:
    def __init__(self):
        self.db = VectorDatabase()

        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.3
        )

        # 🎓 PROMPT РЕПЕТИТОРА
        self.prompt_template = ChatPromptTemplate.from_template("""
Ты — AI-репетитор по техническим дисциплинам.

Твоя задача:
1. Понятно и структурировано объяснить тему
2. Использовать информацию ТОЛЬКО из контекста
3. Если вопрос неполный — объясни базовые понятия
4. Приводи примеры, если это уместно

После основного объяснения:
- задай 2 вопроса для самопроверки
- порекомендуй, что изучить дальше

Контекст:
----------------
{context}
----------------

Вопрос студента:
{question}

Ответ (на русском, развёрнуто):
""")

    def get_answer(self, query: str):
        logger.info(f"🔍 Поиск по запросу: {query}")

        # 1️⃣ RETRIEVAL: берём БОЛЬШЕ контекста
        docs_with_scores = self.db.search_with_score(
            query,
            k=settings.TOP_K_RETRIEVAL
        )

        if not docs_with_scores:
            return (
                "Я не нашёл точного ответа в базе, но давай разберём тему пошагово.",
                {}
            )

        # 2️⃣ Сортируем по релевантности
        docs_with_scores.sort(key=lambda x: x[1])

        # 3️⃣ Берём TOP_N без жёстких фильтров
        selected_docs = docs_with_scores[:settings.TOP_N_RERANK]

        docs = [doc for doc, _ in selected_docs]

        # 4️⃣ Формируем обучающий контекст
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # 5️⃣ Источники (прозрачность)
        sources = {
            doc.metadata["source"]: doc.metadata["title"]
            for doc in docs
        }

        logger.info("🤖 Генерация обучающего ответа...")

        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": context_text,
            "question": query
        })

        return response.content.strip(), sources