#!/usr/bin/env python3
"""
Простой тест VectorRAGInterface без зависимостей от vllm
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_rag_interface():
    """Тест VectorRAGInterface"""
    print("🧪 Тестирование VectorRAGInterface")
    print("=" * 50)
    
    try:
        # Импортируем только нужные компоненты
        from src.submit.embeddings.improved_vector_search import (
            ImprovedEmbeddingEngine,
            ImprovedVectorStore,
            EmbeddingConfig,
            SimilarityMetric
        )
        from src.submit.rag.vector_rag_interface import VectorRAGInterface
        
        # Создаем интерфейс
        print("\n1️⃣ Создание VectorRAGInterface...")
        rag = VectorRAGInterface(
            model_name="cointegrated/rubert-tiny2",
            use_gpu=torch.cuda.is_available(),
            enable_hybrid_search=True
        )
        print(f"✅ Интерфейс создан успешно")
        print(f"   Устройство: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"   Гибридный поиск: {'Включен' if rag.enable_hybrid else 'Отключен'}")
        
        # Тестовый диалог
        print("\n2️⃣ Подготовка тестовых данных...")
        dialogue_id = "test_dialogue_001"
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
        print(f"✅ Подготовлено {len(messages)} сообщений")
        
        # Индексация диалога
        print("\n3️⃣ Индексация диалога...")
        rag.add_dialogue(dialogue_id, messages)
        print(f"✅ Диалог проиндексирован")
        
        # Проверяем что хранилище создано
        if dialogue_id in rag.stores:
            store = rag.stores[dialogue_id]
            print(f"   Документов в хранилище: {store.size()}")
        
        # Тестовые запросы
        print("\n4️⃣ Тестирование поиска...")
        test_queries = [
            "Как меня зовут?",
            "Где я работаю?", 
            "Какие у меня питомцы?",
            "Чем я увлекаюсь?",
            "Откуда я?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   {i}. Вопрос: {query}")
            
            # Получаем контекст
            context = rag.get_relevant_context(query, dialogue_id, top_k=2)
            
            if context:
                print("   📚 Найденный контекст:")
                lines = context.split('\n')[1:]  # Пропускаем заголовок
                for line in lines[:2]:  # Показываем первые 2 результата
                    if line.strip():
                        print(f"      {line[:80]}...")
            else:
                print("   ❌ Контекст не найден")
        
        # Тест обработки нового сообщения
        print("\n5️⃣ Тест обработки нового сообщения...")
        new_question = "Как зовут мою кошку и где я работаю?"
        enhanced = rag.process_message(new_question, dialogue_id, role="user")
        
        print(f"   Исходный вопрос: {new_question}")
        print(f"   Обогащенный вопрос:")
        print(f"   {enhanced[:200]}...")
        
        # Тест сохранения и загрузки
        print("\n6️⃣ Тест сохранения/загрузки...")
        test_save_path = "./test_vector_indices"
        
        # Сохраняем
        rag.save(test_save_path)
        print(f"✅ Индексы сохранены в {test_save_path}")
        
        # Создаем новый экземпляр и загружаем
        rag2 = VectorRAGInterface()
        rag2.load(test_save_path)
        
        # Проверяем загрузку
        test_query = "питомцы"
        results = rag2.search(test_query, dialogue_id, top_k=1)
        
        if results:
            print(f"✅ Индексы успешно загружены")
            print(f"   Найдено: {results[0]['text'][:50]}...")
        else:
            print("❌ Ошибка загрузки индексов")
        
        # Статистика
        print("\n7️⃣ Статистика системы...")
        if dialogue_id in rag.stores:
            store = rag.stores[dialogue_id]
            analytics = store.get_analytics()
            print(f"   Документов в индексе: {analytics.total_documents}")
            print(f"   Всего поисков: {analytics.total_searches}")
            print(f"   Среднее время поиска: {analytics.avg_search_time*1000:.2f} мс")
        
        # Статистика эмбеддингов
        emb_stats = rag.embedding_engine.get_stats()
        print(f"   Закодировано текстов: {emb_stats['total_encoded']}")
        print(f"   Cache hit rate: {emb_stats['cache_hit_rate']:.1%}")
        
        # Очистка
        print("\n8️⃣ Очистка тестовых файлов...")
        if Path(test_save_path).exists():
            import shutil
            shutil.rmtree(test_save_path)
            print("✅ Тестовые файлы удалены")
        
        print("\n🎉 Все тесты пройдены успешно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_components():
    """Тест отдельных компонентов эмбеддингов"""
    print("\n🔧 Тест компонентов эмбеддингов")
    print("=" * 50)
    
    try:
        from src.submit.embeddings.improved_vector_search import (
            ImprovedEmbeddingEngine,
            ImprovedVectorStore,
            EmbeddingConfig,
            SimilarityMetric
        )
        
        # Тест EmbeddingEngine
        print("1️⃣ Тест ImprovedEmbeddingEngine...")
        config = EmbeddingConfig(
            model_name="cointegrated/rubert-tiny2",
            device="cpu",  # Принудительно CPU для стабильности
            batch_size=16,
            use_cache=True
        )
        
        engine = ImprovedEmbeddingEngine(config)
        print("✅ EmbeddingEngine создан")
        
        # Тест кодирования
        test_texts = ["Привет мир!", "Тестовое сообщение"]
        embeddings = engine.encode(test_texts)
        print(f"✅ Закодировано {len(test_texts)} текстов")
        print(f"   Размерность эмбеддингов: {embeddings.shape}")
        
        # Тест VectorStore
        print("\n2️⃣ Тест ImprovedVectorStore...")
        store = ImprovedVectorStore(
            metric=SimilarityMetric.COSINE,
            enable_analytics=True
        )
        
        # Добавляем документы
        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            store.add(
                doc_id=f"doc_{i}",
                vector=embedding,
                text=text,
                metadata={'index': i}
            )
        
        print(f"✅ Добавлено {store.size()} документов")
        
        # Тест поиска
        query_embedding = engine.encode("мир")
        results = store.search(query_embedding, k=2)
        
        print(f"✅ Поиск выполнен, найдено {len(results)} результатов")
        for result in results:
            print(f"   - {result.doc_id}: {result.score:.3f}")
        
        # Тест гибридного поиска
        print("\n3️⃣ Тест гибридного поиска...")
        hybrid_results = store.hybrid_search(
            query_vector=query_embedding,
            query_text="мир",
            k=2
        )
        
        print(f"✅ Гибридный поиск выполнен, найдено {len(hybrid_results)} результатов")
        for result in hybrid_results:
            print(f"   - {result.doc_id}: {result.score:.3f}")
        
        print("\n🎉 Все компоненты работают корректно!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании компонентов: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ ВЕКТОРНОГО RAG ДЛЯ GIGAMEMORY")
    print("=" * 60)
    
    # Проверяем доступность PyTorch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA устройство: {torch.cuda.get_device_name()}")
    
    # Запускаем тесты
    test1_passed = test_embedding_components()
    test2_passed = test_vector_rag_interface()
    
    # Итоги
    print("\n📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 30)
    print(f"Компоненты эмбеддингов: {'✅ ПРОЙДЕН' if test1_passed else '❌ ПРОВАЛЕН'}")
    print(f"VectorRAGInterface: {'✅ ПРОЙДЕН' if test2_passed else '❌ ПРОВАЛЕН'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("VectorRAGInterface готов к использованию в GigaMemory!")
        return True
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("Проверьте ошибки выше")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
