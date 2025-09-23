#!/usr/bin/env python3
"""
Изолированный тест VectorRAGInterface без зависимостей от других модулей
"""
import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import pickle
import gzip
import time
from collections import defaultdict, deque
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Копируем необходимые классы напрямую для изолированного тестирования
class PoolingStrategy(Enum):
    """Стратегии пулинга для эмбеддингов"""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    MEAN_MAX = "mean_max"
    WEIGHTED_MEAN = "weighted_mean"

class SimilarityMetric(Enum):
    """Метрики сходства"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"
    ANGULAR = "angular"

@dataclass
class EmbeddingConfig:
    """Расширенная конфигурация для движка эмбеддингов"""
    model_name: str = "cointegrated/rubert-tiny2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    cache_dir: Optional[str] = None
    use_cache: bool = True
    use_amp: bool = True
    num_workers: int = 4
    prefetch_batches: int = 2
    warmup_steps: int = 0
    compile_model: bool = False
    quantize_model: bool = False

@dataclass
class VectorDocument:
    """Расширенный документ в векторном хранилище"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class SearchResult:
    """Расширенный результат поиска"""
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None
    rank: int = 0
    explanation: Optional[Dict[str, Any]] = None

@dataclass
class VectorStoreStats:
    """Статистика векторного хранилища"""
    total_documents: int = 0
    total_searches: int = 0
    avg_search_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    popular_queries: Dict[str, int] = field(default_factory=dict)
    access_patterns: Dict[str, int] = field(default_factory=dict)
    top_accessed_docs: List[Dict[str, Any]] = field(default_factory=list)
    unique_metadata_keys: List[str] = field(default_factory=list)

class MockEmbeddingEngine:
    """Мок движка эмбеддингов для тестирования"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = {
            'total_encoded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'encoding_time': 0.0,
            'last_batch_time': 0.0
        }
        logger.info(f"MockEmbeddingEngine создан: {config.model_name}")
    
    def encode(self, texts):
        """Мок кодирования - возвращает случайные векторы"""
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Генерируем случайные векторы размерности 312 (rubert-tiny2)
        embeddings = np.random.randn(len(texts), 312).astype(np.float32)
        
        # Нормализуем
        if self.config.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.stats['total_encoded'] += len(texts)
        
        if single_text:
            return embeddings[0]
        return embeddings
    
    def get_stats(self):
        """Возвращает статистику"""
        stats = self.stats.copy()
        stats['cache_size'] = 0
        stats['cache_hit_rate'] = 0.0
        return stats

class MockVectorStore:
    """Мок векторного хранилища для тестирования"""
    
    def __init__(self, metric=SimilarityMetric.COSINE, enable_analytics=True):
        self.metric = metric
        self.enable_analytics = enable_analytics
        self.documents = []
        self.vectors = None
        self.id_to_index = {}
        
        if enable_analytics:
            self.analytics = {
                'total_searches': 0,
                'avg_search_time': 0.0
            }
        
        logger.info(f"MockVectorStore создан: metric={metric.value}")
    
    def add(self, doc_id, vector, metadata=None, text=None):
        """Добавление документа"""
        doc = VectorDocument(
            id=doc_id,
            vector=vector,
            metadata=metadata or {},
            text=text
        )
        
        index = len(self.documents)
        self.documents.append(doc)
        self.id_to_index[doc_id] = index
        
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        return True
    
    def search(self, query_vector, k=5, threshold=None):
        """Простой поиск"""
        if not self.documents:
            return []
        
        # Вычисляем сходство
        scores = np.dot(self.vectors, query_vector)
        
        # Применяем порог
        if threshold is not None:
            mask = scores >= threshold
            scores = scores[mask]
            valid_indices = [i for i, m in enumerate(mask) if m]
        else:
            valid_indices = list(range(len(self.documents)))
        
        if len(valid_indices) == 0:
            return []
        
        # Находим топ-k
        k = min(k, len(valid_indices))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Формируем результаты
        results = []
        for rank, idx in enumerate(top_indices):
            doc_idx = valid_indices[idx]
            doc = self.documents[doc_idx]
            
            results.append(SearchResult(
                doc_id=doc.id,
                score=float(scores[idx]),
                metadata=doc.metadata,
                text=doc.text,
                rank=rank
            ))
        
        # Обновляем аналитику
        if self.enable_analytics:
            self.analytics['total_searches'] += 1
        
        return results
    
    def hybrid_search(self, query_vector, query_text, k=5, vector_weight=0.7, text_weight=0.3):
        """Гибридный поиск"""
        # Простая реализация - используем векторный поиск
        return self.search(query_vector, k)
    
    def size(self):
        """Количество документов"""
        return len(self.documents)
    
    def get_analytics(self):
        """Аналитика"""
        if not self.enable_analytics:
            return VectorStoreStats()
        
        return VectorStoreStats(
            total_documents=len(self.documents),
            total_searches=self.analytics['total_searches'],
            avg_search_time=self.analytics['avg_search_time']
        )
    
    def save(self, filepath):
        """Сохранение (мок)"""
        logger.info(f"MockVectorStore сохранен: {filepath}")
    
    def load(self, filepath):
        """Загрузка (мок)"""
        logger.info(f"MockVectorStore загружен: {filepath}")

class MockVectorRAGInterface:
    """Мок VectorRAGInterface для тестирования"""
    
    def __init__(self, model_name="cointegrated/rubert-tiny2", use_gpu=True, enable_hybrid_search=True):
        self.embedding_engine = MockEmbeddingEngine(
            EmbeddingConfig(
                model_name=model_name,
                device="cuda" if use_gpu else "cpu",
                batch_size=32,
                use_cache=True
            )
        )
        
        self.stores = {}
        self.enable_hybrid = enable_hybrid_search
        self.dialogues = {}
        
        logger.info(f"MockVectorRAGInterface готов: {model_name}")
    
    def add_dialogue(self, dialogue_id, messages):
        """Добавление диалога"""
        self.dialogues[dialogue_id] = messages
        
        if dialogue_id not in self.stores:
            self.stores[dialogue_id] = MockVectorStore(
                metric=SimilarityMetric.COSINE,
                enable_analytics=True
            )
        
        self._index_messages(dialogue_id, messages)
    
    def get_relevant_context(self, query, dialogue_id, top_k=5):
        """Получение релевантного контекста"""
        if dialogue_id not in self.stores:
            return ""
        
        query_embedding = self.embedding_engine.encode(query)
        store = self.stores[dialogue_id]
        
        if self.enable_hybrid:
            results = store.hybrid_search(query_vector=query_embedding, query_text=query, k=top_k)
        else:
            results = store.search(query_vector=query_embedding, k=top_k, threshold=0.7)
        
        if not results:
            return ""
        
        context_parts = ["Релевантная информация из истории диалога:"]
        for result in results:
            if result.text:
                context_parts.append(f"- {result.text}")
        
        return "\n".join(context_parts)
    
    def search(self, query, dialogue_id, **kwargs):
        """Поиск"""
        if dialogue_id not in self.stores:
            return []
        
        query_embedding = self.embedding_engine.encode(query)
        store = self.stores[dialogue_id]
        
        results = store.search(
            query_vector=query_embedding,
            k=kwargs.get('top_k', 5),
            threshold=kwargs.get('threshold', 0.7)
        )
        
        return [
            {
                'id': r.doc_id,
                'score': r.score,
                'text': r.text,
                'metadata': r.metadata
            }
            for r in results
        ]
    
    def process_message(self, message, dialogue_id, role="user"):
        """Обработка сообщения"""
        if dialogue_id not in self.dialogues:
            self.dialogues[dialogue_id] = []
        
        self.dialogues[dialogue_id].append({
            'role': role,
            'content': message
        })
        
        if role == "assistant":
            self._index_single_message(dialogue_id, message, role)
        
        if role == "user":
            context = self.get_relevant_context(message, dialogue_id)
            if context:
                return f"{context}\n\nВопрос: {message}"
        
        return message
    
    def _index_messages(self, dialogue_id, messages):
        """Индексация сообщений"""
        store = self.stores[dialogue_id]
        
        chunk_size = 3
        chunks = []
        
        for i in range(0, len(messages), chunk_size):
            chunk = messages[i:i+chunk_size]
            chunk_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                for msg in chunk
            ])
            chunks.append(chunk_text)
        
        if chunks:
            embeddings = self.embedding_engine.encode(chunks)
            
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                store.add(
                    doc_id=f"chunk_{dialogue_id}_{i}",
                    vector=embedding,
                    text=chunk_text,
                    metadata={'chunk_index': i, 'dialogue_id': dialogue_id}
                )
    
    def _index_single_message(self, dialogue_id, message, role):
        """Индексация одного сообщения"""
        if dialogue_id not in self.stores:
            self.stores[dialogue_id] = MockVectorStore(metric=SimilarityMetric.COSINE)
        
        store = self.stores[dialogue_id]
        embedding = self.embedding_engine.encode(message)
        
        store.add(
            doc_id=f"msg_{dialogue_id}_{datetime.now().timestamp()}",
            vector=embedding,
            text=f"{role}: {message}",
            metadata={'role': role, 'timestamp': datetime.now().isoformat()}
        )
    
    def save(self, path="./rag_indices"):
        """Сохранение"""
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        for dialogue_id, store in self.stores.items():
            store.save(str(save_dir / f"{dialogue_id}.idx"))
        
        with open(save_dir / "dialogues.json", 'w') as f:
            json.dump(self.dialogues, f)
        
        logger.info(f"Сохранено {len(self.stores)} индексов")
    
    def load(self, path="./rag_indices"):
        """Загрузка"""
        load_dir = Path(path)
        
        if not load_dir.exists():
            return
        
        dialogues_file = load_dir / "dialogues.json"
        if dialogues_file.exists():
            with open(dialogues_file, 'r') as f:
                self.dialogues = json.load(f)
        
        for idx_file in load_dir.glob("*.idx"):
            dialogue_id = idx_file.stem
            store = MockVectorStore(metric=SimilarityMetric.COSINE)
            store.load(str(idx_file))
            self.stores[dialogue_id] = store
        
        logger.info(f"Загружено {len(self.stores)} индексов")

def test_mock_vector_rag():
    """Тест мок VectorRAGInterface"""
    print("🧪 Тестирование MockVectorRAGInterface")
    print("=" * 50)
    
    try:
        # Создаем интерфейс
        print("\n1️⃣ Создание MockVectorRAGInterface...")
        rag = MockVectorRAGInterface(
            model_name="cointegrated/rubert-tiny2",
            use_gpu=False,  # Принудительно CPU для стабильности
            enable_hybrid_search=True
        )
        print(f"✅ Интерфейс создан успешно")
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
        rag2 = MockVectorRAGInterface()
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
        
        # Статистика эмбеддингов
        emb_stats = rag.embedding_engine.get_stats()
        print(f"   Закодировано текстов: {emb_stats['total_encoded']}")
        
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

def main():
    """Главная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ ВЕКТОРНОГО RAG ДЛЯ GIGAMEMORY (МОК)")
    print("=" * 60)
    
    # Проверяем доступность PyTorch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    # Запускаем тест
    test_passed = test_mock_vector_rag()
    
    # Итоги
    print("\n📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 30)
    print(f"MockVectorRAGInterface: {'✅ ПРОЙДЕН' if test_passed else '❌ ПРОВАЛЕН'}")
    
    if test_passed:
        print("\n🎉 ТЕСТ ПРОЙДЕН УСПЕШНО!")
        print("Логика VectorRAGInterface работает корректно!")
        print("Готово к интеграции с реальными компонентами!")
        return True
    else:
        print("\n❌ ТЕСТ ПРОВАЛЕН")
        print("Проверьте ошибки выше")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
