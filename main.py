import sys
import os
from src.parser import DocumentParser
from src.database import VectorDatabase
from src.rag_engine import RAGEngine
from src.config import settings
from loguru import logger

def build_db():
    parser = DocumentParser(settings.RAW_DATA_PATH)
    data = parser.load_and_parse()
    
    if data:
        db = VectorDatabase()
        db.rebuild(data)
    else:
        logger.error("Не удалось прочитать данные. Проверьте raw_data.txt")

def run_chat():
    if not os.path.exists(settings.DB_PATH):
        logger.warning("⚠️ База данных не найдена. Создаю с нуля...")
        build_db()
    
    rag = RAGEngine()
    print("\n" + "="*50)
    print("🎓 AI-Репетитор готов к работе!")
    print("Команды: 'exit' - выход, 'rebuild' - пересоздать базу")
    print("="*50 + "\n")

    while True:
        query = input("Вы: ")
        if query.lower() in ['exit', 'quit']:
            break
        if query.lower() == 'rebuild':
            build_db()
            rag = RAGEngine() # Перезагрузка движка
            continue
            
        answer, sources = rag.get_answer(query)
        
        print(f"\n🤖 Ответ:\n{answer}")
        
        if sources:
            print("\n📚 Источники:")
            for i, (link, title) in enumerate(sources.items(), 1):
                print(f"{i}. {title}")
                print(f"   🔗 {link}")
        print("-" * 50)

if __name__ == "__main__":
    # Если запустить 'python3 main.py --build', то только пересоздаст базу
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_db()
    else:
        run_chat()