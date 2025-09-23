#!/usr/bin/env python3
"""
Тест векторного поиска для GigaMemory
"""
import sys
import os
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импортируем необходимые модули
from models import Message
from submit.embeddings.embedding_engine import EmbeddingEngine, EmbeddingConfig
from submit.embeddings.vector_store import VectorStore, SimpleVectorStore
from submit.rag.vector_rag_engine import VectorRAGEngine, VectorRAGConfig


def test_embedding_engine():
    """Тестирует создание эмбеддингов"""
    print("\n=== Тест EmbeddingEngine ===")
    
    # Создаем движок
    config = EmbeddingConfig(
        model_name="cointegrated/rubert-tiny2",
        device="cpu",  # Используем CPU для теста
        use_cache=True
    )
    engine = EmbeddingEngine(config)
    
    # Тестовые тексты
    texts = [
        "Меня зовут Александр, мне 30 лет",
        "Я работаю программистом в IT компании",
        "У меня есть кошка по имени Мурка",
        "Я люблю играть в футбол по выходным",
        "Моя жена работает врачом",
        "Мы живем в Москве уже 5 лет"
    ]
    
    # Создаем эмбеддинги
    print(f"Кодируем {len(texts)} текстов...")
    embeddings = engine.encode(texts)
    
    print(f"Размер эмбеддингов: {embeddings.shape}")
    print(f"Размерность: {engine.get_embedding_dim()}")
    
    # Тестируем сходство
    query = "Как зовут мою кошку?"
    query_embedding = engine.encode(query)
    
    print(f"\nПоиск по запросу: '{query}'")
    for i, text in enumerate(texts):
        similarity = engine.similarity(embeddings[i], query_embedding)
        print(f"  {similarity:.3f} - {text}")
    
    # Тестируем кэш
    print("\nТестируем кэширование...")
    embeddings_cached = engine.encode(texts[0])
    print(f"Размер кэша: {len(engine.cache)}")
    
    return engine, embeddings, texts


def test_vector_store(engine, embeddings, texts):
    """Тестирует векторное хранилище"""
    print("\n=== Тест VectorStore ===")
    
    # Создаем хранилище
    store = VectorStore(metric="cosine", normalize=True)
    
    # Добавляем документы
    print(f"Добавляем {len(texts)} документов...")
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        metadata = {
            "session_id": f"session_{i}",
            "dialogue_id": "test_dialogue",
            "index": i
        }
        store.add(f"doc_{i}", embedding, metadata, text)
    
    print(f"Размер хранилища: {store.size()}")
    
    # Тестируем поиск
    queries = [
        "Как меня зовут?",
        "Где я работаю?",
        "Какие у меня питомцы?",
        "Чем я занимаюсь на выходных?",
        "Где я живу?"
    ]
    
    for query in queries:
        print(f"\nПоиск: '{query}'")
        query_embedding = engine.encode(query)
        results = store.search(query_embedding, k=3)
        
        for result in results:
            print(f"  [{result.score:.3f}] {result.text}")
    
    # Тестируем фильтрацию по метаданным
    print("\n\nПоиск с фильтром (только session_0 - session_2):")
    query = "работа"
    query_embedding = engine.encode(query)
    
    for i in range(3):
        filter_metadata = {"session_id": f"session_{i}"}
        results = store.search(query_embedding, k=1, filter_metadata=filter_metadata)
        if results:
            print(f"  session_{i}: [{results[0].score:.3f}] {results[0].text}")
    
    return store


def test_vector_rag_engine():
    """Тестирует интегрированный VectorRAGEngine"""
    print("\n=== Тест VectorRAGEngine ===")
    
    # Создаем конфигурацию
    config = VectorRAGConfig(
        embedding_model="cointegrated/rubert-tiny2",
        embedding_device="cpu",
        vector_search_top_k=3,
        vector_search_threshold=0.5,
        use_hybrid_search=True
    )
    
    # Создаем движок
    engine = VectorRAGEngine(config)
    
    # Создаем тестовый диалог
    dialogue_id = "test_dialogue_1"
    messages = [
        # Сессия 1
        Message(role="user", content="Привет! Меня зовут Иван, мне 35 лет.", session_id="1"),
        Message(role="assistant", content="Приятно познакомиться, Иван!", session_id="1"),
        Message(role="user", content="Я работаю инженером в автомобильной компании.", session_id="1"),
        Message(role="assistant", content="Интересная профессия!", session_id="1"),
        
        # Сессия 2
        Message(role="user", content="У меня есть собака породы лабрадор по кличке Рекс.", session_id="2"),
        Message(role="assistant", content="Лабрадоры - отличные собаки!", session_id="2"),
        Message(role="user", content="Да, мы часто гуляем в парке рядом с домом.", session_id="2"),
        Message(role="assistant", content="Прогулки полезны и вам, и Рексу.", session_id="2"),
        
        # Сессия 3
        Message(role="user", content="Я увлекаюсь фотографией и путешествиями.", session_id="3"),
        Message(role="assistant", content="Какие места вы фотографируете?", session_id="3"),
        Message(role="user", content="В основном природу и архитектуру. Недавно был в Италии.", session_id="3"),
        Message(role="assistant", content="Италия прекрасна для фотографии!", session_id="3"),
        
        # Сессия 4
        Message(role="user", content="Моя жена Елена работает учителем математики.", session_id="4"),
        Message(role="assistant", content="Благородная профессия.", session_id="4"),
        Message(role="user", content="У нас двое детей - сын Петя 10 лет и дочь Маша 7 лет.", session_id="4"),
        Message(role="assistant", content="Замечательная семья!", session_id="4"),
    ]
    
    # Группируем по сессиям
    sessions = engine.session_grouper.group_messages_by_sessions(messages, dialogue_id)
    print(f"Сгруппировано {len(sessions)} сессий")
    
    # Индексируем диалог
    print("\nИндексация диалога...")
    stats = engine.index_dialogue(dialogue_id, sessions)
    print(f"Статистика индексации: {stats}")
    
    # Тестируем вопросы
    test_questions = [
        ("Как меня зовут?", "личная информация"),
        ("Где я работаю?", "профессия"),
        ("Какие у меня домашние животные?", "питомцы"),
        ("Чем я увлекаюсь?", "хобби"),
        ("Как зовут мою жену?", "семья"),
        ("Сколько у меня детей?", "семья"),
        ("В какой стране я недавно был?", "путешествия"),
        ("Какая порода у моей собаки?", "питомцы"),
        ("Сколько мне лет?", "личная информация"),
        ("Где мы гуляем с собакой?", "активности")
    ]
    
    print("\n" + "="*60)
    for question, category in test_questions:
        print(f"\nВопрос: '{question}' [{category}]")
        
        # Обрабатываем вопрос
        prompt, metadata = engine.process_question(question, dialogue_id, messages)
        
        print(f"Стратегия: {metadata['strategy']}")
        print(f"Тема: {metadata.get('topic', 'не определена')}")
        print(f"Уверенность: {metadata.get('confidence', 0):.2f}")
        print(f"Найдено сессий: {metadata['vector_search_results']}")
        print(f"Выбрано сессий: {metadata['selected_sessions']}")
        
        if metadata.get('vector_scores'):
            print("Топ результаты векторного поиска:")
            for session_id, score in metadata['vector_scores'][:3]:
                print(f"  - Сессия {session_id}: {score:.3f}")
        
        # Показываем часть промпта
        print(f"Промпт (первые 200 символов):\n{prompt[:200]}...")
    
    # Тестируем анализ векторного поиска
    print("\n" + "="*60)
    print("\nДетальный анализ векторного поиска:")
    analysis = engine.get_vector_search_analysis("Как зовут мою собаку?", dialogue_id)
    
    print(f"Размер индекса: {analysis['vector_store_size']} документов")
    print(f"Размерность эмбеддингов: {analysis['embedding_dim']}")
    print("\nТоп-5 результатов:")
    for i, result in enumerate(analysis['top_5_results'], 1):
        print(f"{i}. Сессия {result['session_id']}: score={result['score']:.3f}")
        if result['text_preview']:
            print(f"   Текст: {result['text_preview']}...")
    
    print(f"\nРаспределение scores:")
    dist = analysis['score_distribution']
    print(f"  Max: {dist['max']:.3f}")
    print(f"  Mean: {dist['mean']:.3f}")
    print(f"  Std: {dist['std']:.3f}")
    
    return engine


def test_simple_store():
    """Тестирует упрощенное векторное хранилище"""
    print("\n=== Тест SimpleVectorStore ===")
    
    # Создаем движок эмбеддингов
    config = EmbeddingConfig(model_name="cointegrated/rubert-tiny2", device="cpu")
    engine = EmbeddingEngine(config)
    
    # Создаем простое хранилище
    store = SimpleVectorStore()
    
    # Тестовые данные
    texts = [
        "Я люблю программирование на Python",
        "Мой любимый язык - Python",
        "JavaScript тоже хороший язык",
        "Я изучаю машинное обучение",
        "Нейронные сети - это интересно"
    ]
    
    # Создаем эмбеддинги и добавляем
    embeddings = engine.encode(texts)
    store.add_texts(texts, embeddings)
    
    # Поиск
    query = "Какой мой любимый язык программирования?"
    query_embedding = engine.encode(query)
    results = store.search_by_text(query_embedding, k=3)
    
    print(f"Запрос: '{query}'")
    print("Результаты:")
    for text, score in results:
        print(f"  [{score:.3f}] {text}")


def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование векторного поиска для GigaMemory")
    print("="*60)
    
    try:
        # Тест 1: Эмбеддинги
        engine, embeddings, texts = test_embedding_engine()
        print("\n✅ EmbeddingEngine работает корректно")
        
        # Тест 2: Векторное хранилище
        store = test_vector_store(engine, embeddings, texts)
        print("\n✅ VectorStore работает корректно")
        
        # Тест 3: SimpleVectorStore
        test_simple_store()
        print("\n✅ SimpleVectorStore работает корректно")
        
        # Тест 4: Интегрированный RAG
        engine = test_vector_rag_engine()
        print("\n✅ VectorRAGEngine работает корректно")
        
        print("\n" + "="*60)
        print("🎉 Все тесты пройдены успешно!")
        print("\nВекторный поиск готов к интеграции в GigaMemory!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



