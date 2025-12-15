import re
from typing import List, Dict
from loguru import logger

class DocumentParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_parse(self) -> List[Dict[str, str]]:
        logger.info(f"⏳ Начинаю обработку файла: {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            logger.error(f"❌ Файл {self.file_path} не найден!")
            return []

        # ВАША ЛОГИКА: Разделяем по длинным линиям "="
        chunks = re.split(r'={20,}', text)

        knowledge_base = []
        pending_meta = None 

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Проверка заголовка
            if "СТРАНИЦА" in chunk and "URL:" in chunk:
                try:
                    title_match = re.search(r'СТРАНИЦА \d+:\s*(.+)', chunk)
                    title = title_match.group(1).strip() if title_match else "Без названия"

                    url_match = re.search(r'URL:\s*(https?://\S+)', chunk)
                    url = url_match.group(1).strip() if url_match else ""

                    if title and url:
                        pending_meta = {"title": title, "source": url}
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка заголовка: {e}")

            # Проверка контента
            elif pending_meta:
                content = re.sub(r'-{20,}', '', chunk).strip()
                
                if content and len(content) > 50: # Фильтр совсем мусорных кусков
                    knowledge_base.append({
                        "title": pending_meta['title'],
                        "source": pending_meta['source'],
                        "content": content
                    })
                
                pending_meta = None

        logger.success(f"✅ Успешно распарсено страниц: {len(knowledge_base)}")
        return knowledge_base