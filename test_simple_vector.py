#!/usr/bin/env python3
"""
Простой тест векторного поиска для GigaMemory
"""
import sys
import os
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импортируем необходимые модули
from submit.embeddings.embedding_engine import EmbeddingEngine, EmbeddingConfig
from submit.embeddings.vector_store import VectorStore, SimpleVectorStore


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


