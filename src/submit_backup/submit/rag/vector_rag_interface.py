"""
Drop-in замена для интеграции векторного поиска в существующий проект GigaMemory
Просто замените RAGInterface на VectorRAGInterface в вашем коде!
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime

# Импортируем улучшенные компоненты
from ..embeddings.improved_vector_search import (
    ImprovedEmbeddingEngine,
    ImprovedVectorStore,
    EmbeddingConfig,
    SimilarityMetric
)

logger = logging.getLogger(__name__)


class VectorRAGInterface:
    """
    Drop-in замена для RAGInterface с векторным поиском
    Полностью совместима с существующим API
    """
    
    def __init__(self, 
                 model_name: str = "cointegrated/rubert-tiny2",
                 use_gpu: bool = True,
                 enable_hybrid_search: bool = True):
        """
        Инициализация с минимальными параметрами
        
        Args:
            model_name: Модель для эмбеддингов
            use_gpu: Использовать GPU если доступен
            enable_hybrid_search: Включить гибридный поиск
        """
        # Определяем устройство
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        
        # Создаем движок эмбеддингов
        self.embedding_engine = ImprovedEmbeddingEngine(
            EmbeddingConfig(
                model_name=model_name,
                device=device,
                batch_size=32,
                use_cache=True,
                use_amp=device == "cuda"
            )
        )
        
        # Хранилища для каждого диалога
        self.stores = {}
        self.enable_hybrid = enable_hybrid_search
        
        # Совместимость с оригинальным RAGInterface
        self.dialogues = {}  # Для обратной совместимости
        
        logger.info(f"VectorRAGInterface готов: {model_name} на {device}")
    
    def add_dialogue(self, dialogue_id: str, messages: List[Dict[str, str]]):
        """
        Добавление диалога (совместимо с оригинальным API)
        
        Args:
            dialogue_id: ID диалога
            messages: Список сообщений
        """
        # Сохраняем для совместимости
        self.dialogues[dialogue_id] = messages
        
        # Создаем хранилище если нет
        if dialogue_id not in self.stores:
            self.stores[dialogue_id] = ImprovedVectorStore(
                metric=SimilarityMetric.COSINE,
                enable_analytics=True
            )
        
        # Индексируем сообщения
        self._index_messages(dialogue_id, messages)
    
    def get_relevant_context(self, 
                            query: str, 
                            dialogue_id: str,
                            top_k: int = 5) -> str:
        """
        Получение релевантного контекста (совместимо с оригинальным API)
        
        Args:
            query: Запрос
            dialogue_id: ID диалога
            top_k: Количество результатов
            
        Returns:
            Строка с контекстом
        """
        if dialogue_id not in self.stores:
            return ""
        
        # Кодируем запрос
        query_embedding = self.embedding_engine.encode(query)
        
        # Ищем релевантные сообщения
        store = self.stores[dialogue_id]
        
        if self.enable_hybrid:
            results = store.hybrid_search(
                query_vector=query_embedding,
                query_text=query,
                k=top_k
            )
        else:
            results = store.search(
                query_vector=query_embedding,
                k=top_k,
                threshold=0.7
            )
        
        # Формируем контекст
        if not results:
            return ""
        
        context_parts = ["Релевантная информация из истории диалога:"]
        
        for result in results:
            if result.text:
                context_parts.append(f"- {result.text}")
        
        return "\n".join(context_parts)
    
    def search(self, query: str, dialogue_id: str, **kwargs) -> List[Dict]:
        """
        Поиск (совместимо с оригинальным API)
        
        Returns:
            Список результатов в формате оригинального RAGInterface
        """
        if dialogue_id not in self.stores:
            return []
        
        # Используем векторный поиск
        query_embedding = self.embedding_engine.encode(query)
        store = self.stores[dialogue_id]
        
        results = store.search(
            query_vector=query_embedding,
            k=kwargs.get('top_k', 5),
            threshold=kwargs.get('threshold', 0.7)
        )
        
        # Конвертируем в формат оригинального API
        return [
            {
                'id': r.doc_id,
                'score': r.score,
                'text': r.text,
                'metadata': r.metadata
            }
            for r in results
        ]
    
    def process_message(self, 
                       message: str,
                       dialogue_id: str,
                       role: str = "user") -> str:
        """
        Обработка сообщения с автоматической индексацией
        
        Args:
            message: Текст сообщения
            dialogue_id: ID диалога
            role: Роль (user/assistant)
            
        Returns:
            Обогащенное сообщение с контекстом
        """
        # Добавляем сообщение в диалог
        if dialogue_id not in self.dialogues:
            self.dialogues[dialogue_id] = []
        
        self.dialogues[dialogue_id].append({
            'role': role,
            'content': message
        })
        
        # Индексируем если это сообщение ассистента
        if role == "assistant":
            self._index_single_message(dialogue_id, message, role)
        
        # Для вопросов пользователя - ищем контекст
        if role == "user":
            context = self.get_relevant_context(message, dialogue_id)
            if context:
                return f"{context}\n\nВопрос: {message}"
        
        return message
    
    def _index_messages(self, dialogue_id: str, messages: List[Dict[str, str]]):
        """Индексация списка сообщений"""
        store = self.stores[dialogue_id]
        
        # Группируем по 3-5 сообщений для лучшего контекста
        chunk_size = 3
        chunks = []
        
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i+chunk_size]
            chunk_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in chunk
            ])
            chunks.append(chunk_text)
        
        # Кодируем все чанки
        if chunks:
            embeddings = self.embedding_engine.encode(chunks)
            
            # Добавляем в хранилище
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                store.add(
                    doc_id=f"chunk_{dialogue_id}_{i}",
                    vector=embedding,
                    text=chunk_text,
                    metadata={'chunk_index': i, 'dialogue_id': dialogue_id}
                )
    
    def _index_single_message(self, dialogue_id: str, message: str, role: str):
        """Индексация одного сообщения"""
        if dialogue_id not in self.stores:
            self.stores[dialogue_id] = ImprovedVectorStore(
                metric=SimilarityMetric.COSINE
            )
        
        store = self.stores[dialogue_id]
        embedding = self.embedding_engine.encode(message)
        
        store.add(
            doc_id=f"msg_{dialogue_id}_{datetime.now().timestamp()}",
            vector=embedding,
            text=f"{role}: {message}",
            metadata={'role': role, 'timestamp': datetime.now().isoformat()}
        )
    
    def save(self, path: str = "./rag_indices"):
        """Сохранение индексов"""
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        for dialogue_id, store in self.stores.items():
            store.save(str(save_dir / f"{dialogue_id}.idx"))
        
        # Сохраняем диалоги
        with open(save_dir / "dialogues.json", 'w') as f:
            json.dump(self.dialogues, f)
        
        logger.info(f"Сохранено {len(self.stores)} индексов")
    
    def load(self, path: str = "./rag_indices"):
        """Загрузка индексов"""
        load_dir = Path(path)
        
        if not load_dir.exists():
            return
        
        # Загружаем диалоги
        dialogues_file = load_dir / "dialogues.json"
        if dialogues_file.exists():
            with open(dialogues_file, 'r') as f:
                self.dialogues = json.load(f)
        
        # Загружаем индексы
        for idx_file in load_dir.glob("*.idx"):
            dialogue_id = idx_file.stem
            store = ImprovedVectorStore(metric=SimilarityMetric.COSINE)
            store.load(str(idx_file))
            self.stores[dialogue_id] = store
        
        logger.info(f"Загружено {len(self.stores)} индексов")


# ============== Простая интеграция в существующий код ==============

def integrate_into_gigamemory():
    """
    Пример интеграции в существующий файл model_inference.py
    """
    code = '''
# В файле src/submit/model_inference.py замените:

# БЫЛО:
# from .rag.rag_engine import RAGInterface

# СТАЛО:
from .rag.vector_rag_interface import VectorRAGInterface

class SubmitModelWithMemory(ModelWithMemory):
    def __init__(self, model_path: str) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        
        # БЫЛО:
        # self.rag_interface = RAGInterface()
        
        # СТАЛО:
        self.rag_interface = VectorRAGInterface(
            model_name="cointegrated/rubert-tiny2",  # Или другая модель
            use_gpu=True,  # Использовать GPU если доступен
            enable_hybrid_search=True  # Гибридный поиск
        )
    
    def update_dialogue(self, dialogue_id: str, messages: List[DialogueUtterance]) -> None:
        """Обновление диалога с векторной индексацией"""
        # Конвертируем в нужный формат
        formatted_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in messages
        ]
        
        # Добавляем в RAG (автоматическая векторизация)
        self.rag_interface.add_dialogue(dialogue_id, formatted_messages)
        
        # Сохраняем в storage как обычно
        self.storage.save_dialogue(dialogue_id, messages)
    
    def ask(self, 
            question: str, 
            dialogue_id: str, 
            model_params: Dict[str, Any]) -> str:
        """Ответ на вопрос с использованием векторного поиска"""
        
        # Получаем релевантный контекст через векторный поиск
        context = self.rag_interface.get_relevant_context(
            question, 
            dialogue_id,
            top_k=5  # Топ-5 релевантных фрагментов
        )
        
        # Формируем промпт с контекстом
        if context:
            enhanced_prompt = f"{context}\\n\\nВопрос: {question}"
        else:
            enhanced_prompt = question
        
        # Генерируем ответ
        response = self.model_inference.generate(
            enhanced_prompt,
            **model_params
        )
        
        # Индексируем ответ для будущего использования
        self.rag_interface.process_message(
            response,
            dialogue_id,
            role="assistant"
        )
        
        return response
'''
    return code


# ============== Тестовый скрипт ==============

def test_integration():
    """
    Тест интеграции векторного поиска
    """
    print("🧪 Тестирование векторного RAG для GigaMemory")
    print("=" * 60)
    
    # 1. Создаем интерфейс
    rag = VectorRAGInterface(
        model_name="cointegrated/rubert-tiny2",
        use_gpu=torch.cuda.is_available(),
        enable_hybrid_search=True
    )
    
    # 2. Тестовый диалог
    dialogue_id = "test_dialogue"
    messages = [
        {"role": "user", "content": "Привет! Меня зовут Алексей, я из Москвы."},
        {"role": "assistant", "content": "Здравствуйте, Алексей! Рад познакомиться. Как дела в Москве?"},
        {"role": "user", "content": "Отлично! Я работаю дата-сайентистом в Сбере."},
        {"role": "assistant", "content": "Замечательно! Data Science - очень интересная область."},
        {"role": "user", "content": "У меня есть кошка Мурка и собака Рекс."},
        {"role": "assistant", "content": "Здорово! Мурка и Рекс - прекрасные имена для питомцев."},
        {"role": "user", "content": "Я увлекаюсь машинным обучением и нейросетями."},
        {"role": "assistant", "content": "Отличное увлечение! ML и нейросети сейчас очень актуальны."},
    ]
    
    # 3. Индексируем диалог
    print("\n📝 Индексация диалога...")
    rag.add_dialogue(dialogue_id, messages)
    print(f"✅ Проиндексировано {len(messages)} сообщений")
    
    # 4. Тестовые запросы
    test_queries = [
        "Как меня зовут?",
        "Где я работаю?",
        "Какие у меня питомцы?",
        "Чем я увлекаюсь?",
        "Откуда я?"
    ]
    
    print("\n🔍 Тестирование векторного поиска:")
    for query in test_queries:
        print(f"\n❓ Вопрос: {query}")
        context = rag.get_relevant_context(query, dialogue_id, top_k=2)
        
        if context:
            print("📚 Найденный контекст:")
            for line in context.split('\n')[1:3]:  # Показываем первые 2 результата
                if line.strip():
                    print(f"  {line[:100]}...")
        else:
            print("  ❌ Контекст не найден")
    
    # 5. Тест обработки нового сообщения
    print("\n💬 Обработка нового вопроса:")
    new_question = "Как зовут мою кошку и где я работаю?"
    enhanced = rag.process_message(new_question, dialogue_id, role="user")
    print(f"Обогащенный вопрос:\n{enhanced[:300]}...")
    
    # 6. Сохранение и загрузка
    print("\n💾 Тест сохранения/загрузки...")
    rag.save("./test_indices")
    
    # Создаем новый экземпляр и загружаем
    rag2 = VectorRAGInterface()
    rag2.load("./test_indices")
    
    # Проверяем что загрузилось
    test_query = "питомцы"
    results = rag2.search(test_query, dialogue_id, top_k=1)
    
    if results:
        print(f"✅ Индексы успешно загружены. Найдено: {results[0]['text'][:50]}...")
    else:
        print("❌ Ошибка загрузки индексов")
    
    # 7. Статистика
    print("\n📊 Статистика:")
    if dialogue_id in rag.stores:
        store = rag.stores[dialogue_id]
        analytics = store.get_analytics()
        print(f"  Документов в индексе: {analytics.get('total_documents', 0)}")
        print(f"  Всего поисков: {analytics.get('total_searches', 0)}")
        print(f"  Среднее время поиска: {analytics.get('avg_search_time', 0)*1000:.2f} мс")
    
    # Статистика эмбеддингов
    emb_stats = rag.embedding_engine.get_stats()
    print(f"  Закодировано текстов: {emb_stats.get('total_encoded', 0)}")
    print(f"  Cache hit rate: {emb_stats.get('cache_hit_rate', 0):.1%}")
    
    print("\n✅ Все тесты пройдены успешно!")
    
    # Очистка
    import shutil
    if Path("./test_indices").exists():
        shutil.rmtree("./test_indices")


# ============== Инструкция по установке ==============

def print_installation_guide():
    """
    Выводит инструкцию по установке
    """
    guide = '''
📦 ИНСТРУКЦИЯ ПО УСТАНОВКЕ ВЕКТОРНОГО ПОИСКА В GIGAMEMORY
=========================================================

1️⃣ КОПИРОВАНИЕ ФАЙЛОВ:
   ```bash
   # Создайте структуру папок
   mkdir -p src/submit/embeddings
   mkdir -p src/submit/rag
   
   # Скопируйте файлы:
   - improved_vector_search.py → src/submit/embeddings/
   - vector_rag_integration.py → src/submit/rag/
   - vector_rag_interface.py → src/submit/rag/
   ```

2️⃣ ОБНОВЛЕНИЕ ЗАВИСИМОСТЕЙ:
   Все нужные библиотеки уже установлены:
   ✅ torch
   ✅ transformers
   ✅ numpy

3️⃣ ИНТЕГРАЦИЯ В КОД:
   
   Вариант A: Полная замена (рекомендуется)
   -----------------------------------------
   В файле src/submit/model_inference.py:
   
   ```python
   # Замените импорт
   from .rag.vector_rag_interface import VectorRAGInterface
   
   # В __init__ замените
   self.rag_interface = VectorRAGInterface(
       model_name="cointegrated/rubert-tiny2",
       use_gpu=True,
       enable_hybrid_search=True
   )
   ```
   
   Вариант B: Параллельное использование
   --------------------------------------
   ```python
   def __init__(self, model_path: str, use_vector_rag: bool = True):
       if use_vector_rag:
           from .rag.vector_rag_interface import VectorRAGInterface
           self.rag_interface = VectorRAGInterface()
       else:
           self.rag_interface = RAGInterface()  # Старый
   ```

4️⃣ ТЕСТИРОВАНИЕ:
   ```bash
   python -c "from src.submit.rag.vector_rag_interface import test_integration; test_integration()"
   ```

5️⃣ ВЫБОР МОДЕЛИ:
   
   Для русского языка (рекомендуемые):
   - "cointegrated/rubert-tiny2" (312MB, быстрая) ⭐
   - "cointegrated/LaBSE-en-ru" (490MB, качественная)
   - "ai-forever/sbert_large_nlu_ru" (1.7GB, точная)
   
   Мультиязычные:
   - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
   - "intfloat/multilingual-e5-base"

6️⃣ НАСТРОЙКА ПРОИЗВОДИТЕЛЬНОСТИ:
   
   Для больших объемов (>10k документов):
   ```python
   config = VectorRAGConfig(
       embedding_batch_size=64,  # Больше батч
       use_amp=True,             # Mixed precision
       compile_model=True,       # torch.compile
       optimize_on_save=True     # Оптимизация при сохранении
   )
   ```

7️⃣ МОНИТОРИНГ:
   ```python
   # Получение статистики
   stats = rag_interface.embedding_engine.get_stats()
   print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
   ```

✅ ГОТОВО К ИСПОЛЬЗОВАНИЮ!

При проблемах проверьте:
- Версия transformers >= 4.30.0
- Версия torch >= 1.10.0
- Достаточно памяти для модели (min 2GB RAM)
    '''
    return guide


if __name__ == "__main__":
    print("\n🚀 ВЕКТОРНЫЙ ПОИСК ДЛЯ GIGAMEMORY\n")
    print("Выберите действие:")
    print("1. Запустить тесты")
    print("2. Показать инструкцию по установке")
    print("3. Показать код для интеграции")
    
    choice = input("\nВаш выбор (1/2/3): ").strip()
    
    if choice == "1":
        test_integration()
    elif choice == "2":
        print(print_installation_guide())
    elif choice == "3":
        print(integrate_into_gigamemory())
    else:
        print("Неверный выбор. Запускаем тесты по умолчанию...")
        test_integration()
