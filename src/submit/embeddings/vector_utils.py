"""
Утилиты для векторного поиска и бенчмарки производительности
"""
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .improved_vector_search import ImprovedEmbeddingEngine, EmbeddingConfig
from .improved_vector_store import ImprovedVectorStore
from .vector_models import SimilarityMetric, PerformanceMetrics

logger = logging.getLogger(__name__)


def benchmark_performance(
    n_documents: int = 1000,
    model_name: str = "cointegrated/rubert-tiny2",
    batch_size: int = 32,
    k: int = 10,
    enable_amp: bool = True,
    use_cache: bool = True
) -> PerformanceMetrics:
    """
    Бенчмарк производительности векторного поиска
    
    Args:
        n_documents: Количество тестовых документов
        model_name: Название модели
        batch_size: Размер батча
        k: Количество результатов поиска
        enable_amp: Использовать AMP
        use_cache: Использовать кэш
    
    Returns:
        Метрики производительности
    """
    print("🚀 Бенчмарк производительности векторного поиска")
    print("=" * 60)
    
    # Создаем движок и хранилище
    config = EmbeddingConfig(
        model_name=model_name,
        batch_size=batch_size,
        use_cache=use_cache,
        use_amp=enable_amp
    )
    
    engine = ImprovedEmbeddingEngine(config)
    store = ImprovedVectorStore(
        metric=SimilarityMetric.COSINE,
        enable_analytics=True
    )
    
    # Генерируем тестовые данные
    texts = [f"Это тестовый документ номер {i} с различным содержанием" for i in range(n_documents)]
    
    # Замеряем время кодирования
    start = time.time()
    embeddings = engine.encode(texts, show_progress=True)
    encoding_time = time.time() - start
    
    encoding_speed = n_documents / encoding_time
    
    print(f"\n📊 Кодирование {n_documents} документов:")
    print(f"   Время: {encoding_time:.2f} сек")
    print(f"   Скорость: {encoding_speed:.0f} док/сек")
    print(f"   Статистика: {engine.get_stats()}")
    
    # Добавляем в хранилище
    start = time.time()
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        store.add(
            doc_id=f"doc_{i}",
            vector=embedding,
            metadata={"category": i % 10},
            text=text
        )
    indexing_time = time.time() - start
    
    indexing_speed = n_documents / indexing_time
    
    print(f"\n📊 Индексация {n_documents} документов:")
    print(f"   Время: {indexing_time:.2f} сек")
    print(f"   Скорость: {indexing_speed:.0f} док/сек")
    
    # Тестируем поиск
    query_text = "документ с содержанием"
    query_embedding = engine.encode(query_text)
    
    # Обычный поиск
    search_times = []
    for _ in range(10):  # Несколько итераций для стабильности
        start = time.time()
        results = store.search(query_embedding, k=k)
        search_time = time.time() - start
        search_times.append(search_time)
    
    avg_search_time = np.mean(search_times)
    search_speed = 1.0 / avg_search_time
    
    print(f"\n📊 Векторный поиск (top-{k}):")
    print(f"   Среднее время: {avg_search_time*1000:.2f} мс")
    print(f"   Скорость: {search_speed:.0f} поисков/сек")
    print(f"   Топ-3 результата:")
    for r in results[:3]:
        print(f"     - {r.doc_id}: {r.score:.3f}")
    
    # Гибридный поиск
    hybrid_times = []
    for _ in range(5):  # Меньше итераций для гибридного поиска
        start = time.time()
        hybrid_results = store.hybrid_search(
            query_embedding, 
            query_text,
            k=k
        )
        hybrid_time = time.time() - start
        hybrid_times.append(hybrid_time)
    
    avg_hybrid_time = np.mean(hybrid_times)
    hybrid_speed = 1.0 / avg_hybrid_time
    
    print(f"\n📊 Гибридный поиск (векторный + текстовый):")
    print(f"   Среднее время: {avg_hybrid_time*1000:.2f} мс")
    print(f"   Скорость: {hybrid_speed:.0f} поисков/сек")
    print(f"   Топ-3 результата:")
    for r in hybrid_results[:3]:
        print(f"     - {r.doc_id}: {r.score:.3f}")
    
    # Аналитика
    print(f"\n📊 Аналитика хранилища:")
    analytics = store.get_analytics()
    print(f"   Всего документов: {analytics.total_documents}")
    print(f"   Всего поисков: {analytics.total_searches}")
    print(f"   Среднее время поиска: {analytics.avg_search_time:.3f} сек")
    print(f"   Уникальных ключей метаданных: {len(analytics.unique_metadata_keys)}")
    
    # Вычисляем метрики производительности
    metrics = PerformanceMetrics(
        encoding_speed=encoding_speed,
        search_speed=search_speed,
        memory_efficiency=n_documents / (store.vectors.nbytes / 1024 / 1024),  # док/МБ
        cache_efficiency=engine.get_stats().get('cache_hit_rate', 0.0),
        accuracy_score=0.0,  # Требует ground truth данных
        latency_p95=np.percentile(search_times, 95) * 1000,  # мс
        throughput=min(encoding_speed, search_speed)  # общая пропускная способность
    )
    
    print(f"\n📊 Итоговые метрики:")
    print(f"   Скорость кодирования: {metrics.encoding_speed:.0f} док/сек")
    print(f"   Скорость поиска: {metrics.search_speed:.0f} поисков/сек")
    print(f"   Эффективность памяти: {metrics.memory_efficiency:.0f} док/МБ")
    print(f"   Эффективность кэша: {metrics.cache_efficiency:.1%}")
    print(f"   P95 задержка: {metrics.latency_p95:.1f} мс")
    print(f"   Общая пропускная способность: {metrics.throughput:.0f} оп/сек")
    
    print("\n✅ Бенчмарк завершен!")
    
    return metrics


def compare_models(
    models: List[str],
    n_documents: int = 100,
    batch_size: int = 16
) -> Dict[str, PerformanceMetrics]:
    """
    Сравнение производительности разных моделей
    
    Args:
        models: Список названий моделей
        n_documents: Количество тестовых документов
        batch_size: Размер батча
    
    Returns:
        Словарь с метриками для каждой модели
    """
    print("🔍 Сравнение моделей для векторного поиска")
    print("=" * 60)
    
    results = {}
    
    for model_name in models:
        print(f"\n🧪 Тестируем модель: {model_name}")
        print("-" * 40)
        
        try:
            metrics = benchmark_performance(
                n_documents=n_documents,
                model_name=model_name,
                batch_size=batch_size,
                enable_amp=True,
                use_cache=True
            )
            results[model_name] = metrics
            
        except Exception as e:
            print(f"❌ Ошибка с моделью {model_name}: {e}")
            continue
    
    # Сравнительная таблица
    print(f"\n📊 Сравнительная таблица:")
    print("-" * 80)
    print(f"{'Модель':<30} {'Кодирование':<12} {'Поиск':<10} {'Память':<10} {'Кэш':<8}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics.encoding_speed:>8.0f} док/с {metrics.search_speed:>6.0f} оп/с "
              f"{metrics.memory_efficiency:>6.0f} док/МБ {metrics.cache_efficiency:>6.1%}")
    
    return results


def stress_test(
    n_documents: int = 10000,
    n_searches: int = 1000,
    concurrent_searches: int = 10
) -> Dict[str, Any]:
    """
    Стресс-тест векторного поиска
    
    Args:
        n_documents: Количество документов
        n_searches: Количество поисковых запросов
        concurrent_searches: Количество одновременных поисков
    
    Returns:
        Результаты стресс-теста
    """
    print("💪 Стресс-тест векторного поиска")
    print("=" * 60)
    
    # Создаем движок и хранилище
    config = EmbeddingConfig(
        model_name="cointegrated/rubert-tiny2",
        batch_size=64,
        use_cache=True,
        use_amp=True
    )
    
    engine = ImprovedEmbeddingEngine(config)
    store = ImprovedVectorStore(
        metric=SimilarityMetric.COSINE,
        enable_analytics=True
    )
    
    # Генерируем большое количество документов
    print(f"📝 Генерируем {n_documents} документов...")
    texts = [f"Документ {i} с уникальным содержанием и различными характеристиками" 
             for i in range(n_documents)]
    
    start = time.time()
    embeddings = engine.encode(texts, show_progress=True)
    encoding_time = time.time() - start
    
    # Добавляем в хранилище
    print(f"📚 Индексируем документы...")
    start = time.time()
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        store.add(
            doc_id=f"stress_doc_{i}",
            vector=embedding,
            metadata={"category": i % 100, "priority": i % 10},
            text=text
        )
    indexing_time = time.time() - start
    
    # Генерируем поисковые запросы
    print(f"🔍 Генерируем {n_searches} поисковых запросов...")
    query_texts = [f"поиск {i} с различными параметрами" for i in range(n_searches)]
    query_embeddings = engine.encode(query_texts, show_progress=True)
    
    # Стресс-тест поиска
    print(f"⚡ Запускаем стресс-тест ({n_searches} поисков)...")
    
    search_times = []
    start = time.time()
    
    for i, (query_text, query_embedding) in enumerate(zip(query_texts, query_embeddings)):
        search_start = time.time()
        
        # Чередуем разные типы поиска
        if i % 3 == 0:
            results = store.search(query_embedding, k=10)
        elif i % 3 == 1:
            results = store.search(query_embedding, k=10, filter_metadata={"category": i % 100})
        else:
            results = store.hybrid_search(query_embedding, query_text, k=10)
        
        search_time = time.time() - search_start
        search_times.append(search_time)
        
        if (i + 1) % 100 == 0:
            print(f"   Обработано {i + 1}/{n_searches} поисков")
    
    total_time = time.time() - start
    
    # Анализ результатов
    results = {
        'total_documents': n_documents,
        'total_searches': n_searches,
        'total_time': total_time,
        'avg_search_time': np.mean(search_times),
        'min_search_time': np.min(search_times),
        'max_search_time': np.max(search_times),
        'p95_search_time': np.percentile(search_times, 95),
        'p99_search_time': np.percentile(search_times, 99),
        'searches_per_second': n_searches / total_time,
        'encoding_time': encoding_time,
        'indexing_time': indexing_time,
        'memory_usage_mb': store.vectors.nbytes / 1024 / 1024,
        'analytics': store.get_analytics()
    }
    
    print(f"\n📊 Результаты стресс-теста:")
    print(f"   Всего документов: {results['total_documents']:,}")
    print(f"   Всего поисков: {results['total_searches']:,}")
    print(f"   Общее время: {results['total_time']:.2f} сек")
    print(f"   Поисков в секунду: {results['searches_per_second']:.0f}")
    print(f"   Среднее время поиска: {results['avg_search_time']*1000:.2f} мс")
    print(f"   P95 время поиска: {results['p95_search_time']*1000:.2f} мс")
    print(f"   P99 время поиска: {results['p99_search_time']*1000:.2f} мс")
    print(f"   Использование памяти: {results['memory_usage_mb']:.1f} МБ")
    
    print("\n✅ Стресс-тест завершен!")
    
    return results


def optimize_store_config(
    store: ImprovedVectorStore,
    test_queries: List[np.ndarray],
    optimization_target: str = "speed"
) -> Dict[str, Any]:
    """
    Оптимизация конфигурации хранилища
    
    Args:
        store: Векторное хранилище
        test_queries: Тестовые запросы
        optimization_target: Цель оптимизации ("speed", "accuracy", "memory")
    
    Returns:
        Результаты оптимизации
    """
    print(f"⚙️ Оптимизация хранилища (цель: {optimization_target})")
    print("=" * 60)
    
    # Тестируем разные конфигурации
    configs = [
        {"k": 5, "rerank": False},
        {"k": 10, "rerank": False},
        {"k": 5, "rerank": True},
        {"k": 10, "rerank": True},
    ]
    
    results = {}
    
    for config in configs:
        config_name = f"k={config['k']}, rerank={config['rerank']}"
        print(f"\n🧪 Тестируем конфигурацию: {config_name}")
        
        search_times = []
        for query in test_queries[:10]:  # Тестируем на первых 10 запросах
            start = time.time()
            results_list = store.search(query, **config)
            search_time = time.time() - start
            search_times.append(search_time)
        
        avg_time = np.mean(search_times)
        results[config_name] = {
            'avg_search_time': avg_time,
            'searches_per_second': 1.0 / avg_time,
            'config': config
        }
        
        print(f"   Среднее время: {avg_time*1000:.2f} мс")
        print(f"   Поисков в секунду: {1.0/avg_time:.0f}")
    
    # Выбираем лучшую конфигурацию
    if optimization_target == "speed":
        best_config = min(results.items(), key=lambda x: x[1]['avg_search_time'])
    else:
        # Для других целей можно добавить дополнительные метрики
        best_config = min(results.items(), key=lambda x: x[1]['avg_search_time'])
    
    print(f"\n🏆 Лучшая конфигурация: {best_config[0]}")
    print(f"   Время поиска: {best_config[1]['avg_search_time']*1000:.2f} мс")
    print(f"   Скорость: {best_config[1]['searches_per_second']:.0f} поисков/сек")
    
    return {
        'best_config': best_config[1]['config'],
        'best_performance': best_config[1],
        'all_results': results
    }


if __name__ == "__main__":
    # Запускаем основной бенчмарк
    benchmark_performance()
    
    # Дополнительно можно запустить сравнение моделей
    # models = ["cointegrated/rubert-tiny2", "DeepPavlov/rubert-base-cased"]
    # compare_models(models, n_documents=50)
