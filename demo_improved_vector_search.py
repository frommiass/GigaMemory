#!/usr/bin/env python3
"""
Демонстрация улучшенной системы векторного поиска
"""
import numpy as np
from src.submit.embeddings import (
    ImprovedEmbeddingEngine, ImprovedVectorStore, EmbeddingConfig,
    SimilarityMetric, PoolingStrategy, benchmark_performance
)


def demo_basic_usage():
    """Демонстрация базового использования"""
    print("🎯 Демонстрация базового использования")
    print("=" * 50)
    
    # Создаем движок эмбеддингов
    config = EmbeddingConfig(
        model_name="cointegrated/rubert-tiny2",
        batch_size=16,
        use_cache=True,
        pooling_strategy=PoolingStrategy.MEAN
    )
    
    engine = ImprovedEmbeddingEngine(config)
    
    # Создаем векторное хранилище
    store = ImprovedVectorStore(
        metric=SimilarityMetric.COSINE,
        enable_analytics=True
    )
    
    # Подготавливаем тестовые данные
    documents = [
        "Машинное обучение - это область искусственного интеллекта",
        "Нейронные сети используются для распознавания образов",
        "Глубокое обучение основано на многослойных нейросетях",
        "Трансформеры революционизировали обработку естественного языка",
        "BERT - это предобученная модель для понимания текста",
        "GPT модели способны генерировать связный текст",
        "Векторные представления слов используются в NLP",
        "Эмбеддинги позволяют представлять текст в числовом виде"
    ]
    
    print(f"📚 Добавляем {len(documents)} документов...")
    
    # Кодируем документы
    embeddings = engine.encode(documents, show_progress=True)
    
    # Добавляем в хранилище
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        store.add(
            doc_id=f"doc_{i}",
            vector=embedding,
            metadata={"category": "AI", "index": i},
            text=doc
        )
    
    print(f"✅ Добавлено {len(documents)} документов")
    
    # Тестируем поиск
    query = "Что такое нейронные сети?"
    print(f"\n🔍 Поиск по запросу: '{query}'")
    
    query_embedding = engine.encode(query)
    results = store.search(query_embedding, k=3)
    
    print("📋 Результаты поиска:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.doc_id}: {result.score:.3f}")
        print(f"     Текст: {result.text}")
        print(f"     Метаданные: {result.metadata}")
        print()
    
    # Гибридный поиск
    print("🔍 Гибридный поиск (векторный + текстовый):")
    hybrid_results = store.hybrid_search(
        query_embedding, 
        query,
        k=3,
        vector_weight=0.7,
        text_weight=0.3
    )
    
    print("📋 Результаты гибридного поиска:")
    for i, result in enumerate(hybrid_results, 1):
        print(f"  {i}. {result.doc_id}: {result.score:.3f}")
        print(f"     Объяснение: {result.explanation}")
        print()
    
    # Статистика
    print("📊 Статистика движка:")
    engine_stats = engine.get_stats()
    for key, value in engine_stats.items():
        print(f"  {key}: {value}")
    
    print("\n📊 Статистика хранилища:")
    store_stats = store.get_analytics()
    print(f"  Всего документов: {store_stats.total_documents}")
    print(f"  Всего поисков: {store_stats.total_searches}")
    print(f"  Среднее время поиска: {store_stats.avg_search_time:.3f} сек")


def demo_advanced_features():
    """Демонстрация продвинутых возможностей"""
    print("\n🚀 Демонстрация продвинутых возможностей")
    print("=" * 50)
    
    # Создаем движок с продвинутыми настройками
    config = EmbeddingConfig(
        model_name="cointegrated/rubert-tiny2",
        batch_size=32,
        use_cache=True,
        use_amp=True,
        pooling_strategy=PoolingStrategy.WEIGHTED_MEAN,
        warmup_steps=2
    )
    
    engine = ImprovedEmbeddingEngine(config)
    
    # Создаем хранилище с разными метриками
    stores = {
        "cosine": ImprovedVectorStore(metric=SimilarityMetric.COSINE),
        "euclidean": ImprovedVectorStore(metric=SimilarityMetric.EUCLIDEAN),
        "angular": ImprovedVectorStore(metric=SimilarityMetric.ANGULAR)
    }
    
    # Тестовые данные
    texts = [
        "Кот сидит на коврике",
        "Собака играет во дворе", 
        "Птица поет на дереве",
        "Рыба плавает в аквариуме",
        "Лошадь бегает по полю"
    ]
    
    embeddings = engine.encode(texts)
    
    # Добавляем в разные хранилища
    for metric_name, store in stores.items():
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            store.add(
                doc_id=f"{metric_name}_doc_{i}",
                vector=embedding,
                metadata={"animal": text.split()[0]},
                text=text
            )
    
    # Тестируем разные метрики
    query = "домашние животные"
    query_embedding = engine.encode(query)
    
    print(f"🔍 Поиск по запросу: '{query}'")
    print("Сравнение разных метрик сходства:")
    
    for metric_name, store in stores.items():
        results = store.search(query_embedding, k=3)
        print(f"\n📊 {metric_name.upper()} метрика:")
        for result in results:
            print(f"  - {result.doc_id}: {result.score:.3f} ({result.text})")
    
    # Фильтрация по метаданным
    print(f"\n🔍 Поиск с фильтрацией по метаданным:")
    cosine_store = stores["cosine"]
    
    # Ищем только документы с определенным животным
    filtered_results = cosine_store.search(
        query_embedding, 
        k=5,
        filter_metadata={"animal": "Кот"}
    )
    
    print("Результаты с фильтром 'animal: Кот':")
    for result in filtered_results:
        print(f"  - {result.doc_id}: {result.score:.3f}")
    
    # Переранжирование
    print(f"\n🔍 Поиск с переранжированием:")
    reranked_results = cosine_store.search(
        query_embedding,
        k=5,
        rerank=True,
        return_explanations=True
    )
    
    print("Результаты с переранжированием:")
    for result in reranked_results:
        print(f"  - {result.doc_id}: {result.score:.3f}")
        if result.explanation:
            print(f"    Объяснение: {result.explanation}")


def demo_performance():
    """Демонстрация производительности"""
    print("\n⚡ Демонстрация производительности")
    print("=" * 50)
    
    # Запускаем бенчмарк
    metrics = benchmark_performance(
        n_documents=500,
        batch_size=32,
        k=10,
        enable_amp=True,
        use_cache=True
    )
    
    print(f"\n🏆 Итоговые метрики производительности:")
    print(f"  Скорость кодирования: {metrics.encoding_speed:.0f} документов/сек")
    print(f"  Скорость поиска: {metrics.search_speed:.0f} поисков/сек")
    print(f"  Эффективность памяти: {metrics.memory_efficiency:.0f} документов/МБ")
    print(f"  Эффективность кэша: {metrics.cache_efficiency:.1%}")
    print(f"  P95 задержка: {metrics.latency_p95:.1f} мс")


def main():
    """Основная функция демонстрации"""
    print("🎯 Демонстрация улучшенной системы векторного поиска")
    print("=" * 60)
    
    try:
        # Базовое использование
        demo_basic_usage()
        
        # Продвинутые возможности
        demo_advanced_features()
        
        # Производительность
        demo_performance()
        
        print("\n✅ Демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
