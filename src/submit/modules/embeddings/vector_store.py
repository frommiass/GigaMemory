"""
Улучшенное векторное хранилище с расширенными возможностями
"""
import numpy as np
import hashlib
import json
import pickle
import gzip
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from collections import defaultdict
from datetime import datetime

from .vector_models import (
    VectorDocument, SearchResult, SimilarityMetric,
    VectorStoreStats, SearchQuery, BatchSearchQuery, HybridSearchQuery
)

logger = logging.getLogger(__name__)


class ImprovedVectorStore:
    """Улучшенное векторное хранилище с расширенными возможностями"""
    
    def __init__(self, 
                 dimension: Optional[int] = None,
                 metric: SimilarityMetric = SimilarityMetric.COSINE,
                 normalize: bool = True,
                 max_documents: int = 1000000,
                 enable_analytics: bool = True):
        """
        Инициализация улучшенного хранилища
        
        Args:
            dimension: Размерность векторов
            metric: Метрика сходства
            normalize: Нормализовать векторы
            max_documents: Максимальное количество документов
            enable_analytics: Включить сбор аналитики
        """
        self.dimension = dimension
        self.metric = metric
        self.normalize = normalize and (metric in [SimilarityMetric.COSINE, SimilarityMetric.ANGULAR])
        self.max_documents = max_documents
        self.enable_analytics = enable_analytics
        
        # Хранилища
        self.vectors: Optional[np.ndarray] = None
        self.documents: List[VectorDocument] = []
        self.id_to_index: Dict[str, int] = {}
        
        # Индексы
        self.metadata_index: Dict[str, Dict[Any, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.text_index: Dict[str, List[int]] = defaultdict(list)  # Инвертированный индекс
        
        # Аналитика
        if enable_analytics:
            self.analytics = {
                'total_searches': 0,
                'avg_search_time': 0.0,
                'popular_queries': defaultdict(int),
                'access_patterns': defaultdict(int)
            }
        
        # Кэш последних поисков
        self.search_cache = {}
        self.cache_size = 100
        
        logger.info(f"Создано улучшенное хранилище: metric={metric.value}, max_docs={max_documents}")
    
    def add(self, doc_id: str, vector: np.ndarray, 
            metadata: Optional[Dict] = None,
            text: Optional[str] = None,
            update_if_exists: bool = True) -> bool:
        """
        Добавление документа с расширенными опциями
        
        Args:
            doc_id: ID документа
            vector: Вектор эмбеддинга
            metadata: Метаданные
            text: Исходный текст
            update_if_exists: Обновить если существует
        
        Returns:
            True если документ добавлен/обновлен
        """
        # Проверка лимита
        if len(self.documents) >= self.max_documents and doc_id not in self.id_to_index:
            logger.warning(f"Достигнут лимит документов: {self.max_documents}")
            return False
        
        # Валидация размерности
        if self.dimension is None:
            self.dimension = vector.shape[0]
        elif vector.shape[0] != self.dimension:
            raise ValueError(f"Неверная размерность: {vector.shape[0]} != {self.dimension}")
        
        # Обработка существующего документа
        if doc_id in self.id_to_index:
            if update_if_exists:
                self._update_document(doc_id, vector, metadata, text)
                return True
            else:
                logger.debug(f"Документ {doc_id} уже существует, пропускаем")
                return False
        
        # Нормализация
        if self.normalize:
            vector = vector / (np.linalg.norm(vector) + 1e-8)
        
        # Создание документа
        doc = VectorDocument(
            id=doc_id,
            vector=vector,
            metadata=metadata or {},
            text=text,
            timestamp=datetime.now()
        )
        
        # Добавление в хранилище
        index = len(self.documents)
        self.documents.append(doc)
        self.id_to_index[doc_id] = index
        
        # Обновление векторной матрицы
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        # Индексация
        self._index_document(doc, index)
        
        # Сброс кэша поиска
        self.search_cache.clear()
        
        return True
    
    def _update_document(self, doc_id: str, vector: np.ndarray, 
                        metadata: Optional[Dict], text: Optional[str]):
        """Обновление существующего документа"""
        index = self.id_to_index[doc_id]
        old_doc = self.documents[index]
        
        # Обновляем вектор
        if self.normalize:
            vector = vector / (np.linalg.norm(vector) + 1e-8)
        self.vectors[index] = vector
        
        # Обновляем документ
        new_doc = VectorDocument(
            id=doc_id,
            vector=vector,
            metadata=metadata or old_doc.metadata,
            text=text or old_doc.text,
            timestamp=datetime.now(),
            access_count=old_doc.access_count
        )
        
        # Переиндексация
        self._deindex_document(old_doc, index)
        self.documents[index] = new_doc
        self._index_document(new_doc, index)
        
        logger.debug(f"Документ {doc_id} обновлен")
    
    def _index_document(self, doc: VectorDocument, index: int):
        """Индексация документа"""
        # Индекс метаданных
        for key, value in doc.metadata.items():
            self.metadata_index[key][value].append(index)
        
        # Текстовый индекс
        if doc.text:
            words = set(doc.text.lower().split())
            for word in words:
                self.text_index[word].append(index)
    
    def _deindex_document(self, doc: VectorDocument, index: int):
        """Удаление из индексов"""
        # Удаление из метаданных
        for key, value in doc.metadata.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                self.metadata_index[key][value].remove(index)
        
        # Удаление из текстового индекса
        if doc.text:
            words = set(doc.text.lower().split())
            for word in words:
                if word in self.text_index:
                    self.text_index[word].remove(index)
    
    def search(self, 
              query_vector: np.ndarray,
              k: int = 5,
              filter_metadata: Optional[Dict] = None,
              filter_text: Optional[str] = None,
              threshold: Optional[float] = None,
              rerank: bool = False,
              return_explanations: bool = False) -> List[SearchResult]:
        """
        Расширенный поиск с дополнительными возможностями
        
        Args:
            query_vector: Вектор запроса
            k: Количество результатов
            filter_metadata: Фильтр по метаданным
            filter_text: Фильтр по тексту (keyword search)
            threshold: Минимальный порог релевантности
            rerank: Переранжирование результатов
            return_explanations: Вернуть объяснения релевантности
        
        Returns:
            Список результатов поиска
        """
        if not self.documents:
            return []
        
        start_time = time.time()
        
        # Проверяем кэш
        cache_key = self._get_cache_key(query_vector, k, filter_metadata, filter_text, threshold)
        if cache_key in self.search_cache:
            logger.debug("Использован кэш поиска")
            return self.search_cache[cache_key]
        
        # Нормализация запроса
        if self.normalize:
            query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Получаем валидные индексы с фильтрацией
        valid_indices = self._get_filtered_indices(filter_metadata, filter_text)
        if not valid_indices:
            return []
        
        # Вычисляем сходство
        scores = self._compute_similarity(query_vector, valid_indices)
        
        # Применяем порог
        if threshold is not None:
            mask = scores >= threshold
            valid_indices = [idx for idx, m in zip(valid_indices, mask) if m]
            scores = scores[mask]
        
        if len(valid_indices) == 0:
            return []
        
        # Находим топ-k
        k = min(k, len(valid_indices))
        if k < len(valid_indices):
            # Используем argpartition для эффективности
            top_indices_local = np.argpartition(scores, -k)[-k:]
        else:
            top_indices_local = np.arange(len(scores))
        
        # Сортируем по scores
        top_indices_local = top_indices_local[np.argsort(scores[top_indices_local])[::-1]]
        
        # Формируем результаты
        results = []
        for rank, idx_local in enumerate(top_indices_local):
            doc_idx = valid_indices[idx_local]
            doc = self.documents[doc_idx]
            
            # Обновляем статистику доступа
            doc.access_count += 1
            doc.last_accessed = datetime.now()
            
            # Создаем результат
            result = SearchResult(
                doc_id=doc.id,
                score=float(scores[idx_local]),
                metadata=doc.metadata.copy(),
                text=doc.text,
                rank=rank
            )
            
            # Добавляем объяснение если нужно
            if return_explanations:
                result.explanation = self._generate_explanation(
                    doc, query_vector, scores[idx_local]
                )
            
            results.append(result)
        
        # Переранжирование если нужно
        if rerank and len(results) > 1:
            results = self._rerank_results(results, query_vector)
        
        # Аналитика
        if self.enable_analytics:
            search_time = time.time() - start_time
            self.analytics['total_searches'] += 1
            self.analytics['avg_search_time'] = (
                (self.analytics['avg_search_time'] * (self.analytics['total_searches'] - 1) + search_time) /
                self.analytics['total_searches']
            )
        
        # Кэширование результата
        if len(self.search_cache) >= self.cache_size:
            # Удаляем старейший элемент
            self.search_cache.pop(next(iter(self.search_cache)))
        self.search_cache[cache_key] = results
        
        return results
    
    def _compute_similarity(self, query_vector: np.ndarray, indices: List[int]) -> np.ndarray:
        """Вычисление сходства с поддержкой разных метрик"""
        vectors_subset = self.vectors[indices]
        
        if self.metric == SimilarityMetric.COSINE:
            # Косинусное сходство (векторы уже нормализованы)
            scores = np.dot(vectors_subset, query_vector)
            
        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            # Скалярное произведение
            scores = np.dot(vectors_subset, query_vector)
            
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Евклидово расстояние (инвертированное)
            distances = np.linalg.norm(vectors_subset - query_vector, axis=1)
            scores = 1.0 / (1.0 + distances)  # Преобразование в сходство
            
        elif self.metric == SimilarityMetric.MANHATTAN:
            # Манхэттенское расстояние
            distances = np.sum(np.abs(vectors_subset - query_vector), axis=1)
            scores = 1.0 / (1.0 + distances)
            
        elif self.metric == SimilarityMetric.ANGULAR:
            # Угловое расстояние
            cos_sim = np.dot(vectors_subset, query_vector)
            cos_sim = np.clip(cos_sim, -1, 1)  # Для численной стабильности
            angles = np.arccos(cos_sim) / np.pi
            scores = 1.0 - angles  # Преобразование в сходство
            
        else:
            raise ValueError(f"Неподдерживаемая метрика: {self.metric}")
        
        return scores
    
    def _get_filtered_indices(self, filter_metadata: Optional[Dict], 
                             filter_text: Optional[str]) -> List[int]:
        """Получение отфильтрованных индексов"""
        valid_indices = None
        
        # Фильтрация по метаданным
        if filter_metadata:
            for key, value in filter_metadata.items():
                if key in self.metadata_index:
                    indices = set(self.metadata_index[key].get(value, []))
                    if valid_indices is None:
                        valid_indices = indices
                    else:
                        valid_indices = valid_indices.intersection(indices)
        
        # Фильтрация по тексту
        if filter_text:
            words = set(filter_text.lower().split())
            text_indices = set()
            
            for word in words:
                if word in self.text_index:
                    if not text_indices:
                        text_indices = set(self.text_index[word])
                    else:
                        text_indices = text_indices.union(self.text_index[word])
            
            if valid_indices is None:
                valid_indices = text_indices
            else:
                valid_indices = valid_indices.intersection(text_indices)
        
        # Если нет фильтров, возвращаем все
        if valid_indices is None:
            return list(range(len(self.documents)))
        
        return list(valid_indices)
    
    def _rerank_results(self, results: List[SearchResult], 
                       query_vector: np.ndarray) -> List[SearchResult]:
        """Переранжирование результатов с учетом дополнительных факторов"""
        # Простая эвристика: учитываем свежесть и популярность
        for result in results:
            doc_idx = self.id_to_index[result.doc_id]
            doc = self.documents[doc_idx]
            
            # Фактор свежести (новые документы получают буст)
            age_days = (datetime.now() - doc.timestamp).days
            freshness_score = 1.0 / (1.0 + age_days / 30.0)
            
            # Фактор популярности
            popularity_score = np.log1p(doc.access_count) / 10.0
            
            # Комбинированный score
            result.score = result.score * 0.7 + freshness_score * 0.2 + popularity_score * 0.1
        
        # Пересортировка
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Обновление рангов
        for i, result in enumerate(results):
            result.rank = i
        
        return results
    
    def _generate_explanation(self, doc: VectorDocument, 
                             query_vector: np.ndarray, score: float) -> Dict[str, Any]:
        """Генерация объяснения релевантности"""
        explanation = {
            'score': float(score),
            'metric': self.metric.value,
            'document_age_days': (datetime.now() - doc.timestamp).days,
            'access_count': doc.access_count
        }
        
        # Анализ по компонентам вектора
        if self.dimension and self.dimension <= 10:
            # Для небольших векторов показываем покомпонентное сходство
            doc_vector = doc.vector
            component_similarity = doc_vector * query_vector
            explanation['component_scores'] = component_similarity.tolist()
        
        return explanation
    
    def _get_cache_key(self, query_vector: np.ndarray, k: int,
                      filter_metadata: Optional[Dict], 
                      filter_text: Optional[str],
                      threshold: Optional[float]) -> str:
        """Генерация ключа кэша для поиска"""
        # Используем хэш вектора и параметров
        vector_hash = hashlib.md5(query_vector.tobytes()).hexdigest()[:8]
        metadata_str = json.dumps(filter_metadata, sort_keys=True) if filter_metadata else ""
        text_str = filter_text or ""
        threshold_str = str(threshold) if threshold else ""
        
        key = f"{vector_hash}_{k}_{metadata_str}_{text_str}_{threshold_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def batch_search(self, query_vectors: np.ndarray, 
                    k: int = 5, **kwargs) -> List[List[SearchResult]]:
        """Батчевый поиск для нескольких запросов"""
        results = []
        for query_vector in query_vectors:
            results.append(self.search(query_vector, k, **kwargs))
        return results
    
    def hybrid_search(self, 
                     query_vector: np.ndarray,
                     query_text: str,
                     k: int = 5,
                     vector_weight: float = 0.7,
                     text_weight: float = 0.3) -> List[SearchResult]:
        """
        Гибридный поиск: векторный + текстовый
        
        Args:
            query_vector: Вектор запроса
            query_text: Текст запроса
            k: Количество результатов
            vector_weight: Вес векторного поиска
            text_weight: Вес текстового поиска
        
        Returns:
            Комбинированные результаты
        """
        # Векторный поиск
        vector_results = self.search(query_vector, k * 2)
        vector_scores = {r.doc_id: r.score for r in vector_results}
        
        # Текстовый поиск
        text_results = self.search(
            query_vector, k * 2, 
            filter_text=query_text
        )
        text_scores = {r.doc_id: r.score for r in text_results}
        
        # Комбинирование scores
        all_doc_ids = set(vector_scores.keys()) | set(text_scores.keys())
        combined_results = []
        
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0)
            t_score = text_scores.get(doc_id, 0)
            combined_score = v_score * vector_weight + t_score * text_weight
            
            # Получаем документ
            doc_idx = self.id_to_index.get(doc_id)
            if doc_idx is not None:
                doc = self.documents[doc_idx]
                
                result = SearchResult(
                    doc_id=doc_id,
                    score=combined_score,
                    metadata=doc.metadata,
                    text=doc.text,
                    explanation={
                        'vector_score': v_score,
                        'text_score': t_score,
                        'combined_score': combined_score
                    }
                )
                combined_results.append(result)
        
        # Сортировка и обрезка
        combined_results.sort(key=lambda x: x.score, reverse=True)
        combined_results = combined_results[:k]
        
        # Обновление рангов
        for i, result in enumerate(combined_results):
            result.rank = i
        
        return combined_results
    
    def get_analytics(self) -> VectorStoreStats:
        """Возвращает аналитику использования"""
        if not self.enable_analytics:
            return VectorStoreStats()
        
        analytics = self.analytics.copy()
        
        # Добавляем статистику по документам
        stats = VectorStoreStats(
            total_documents=len(self.documents),
            total_searches=analytics['total_searches'],
            avg_search_time=analytics['avg_search_time'],
            unique_metadata_keys=list(self.metadata_index.keys())
        )
        
        # Статистика по доступу
        if self.documents:
            access_counts = [doc.access_count for doc in self.documents]
            stats.avg_access_count = np.mean(access_counts)
            stats.max_access_count = max(access_counts)
            
            # Топ-5 популярных документов
            top_docs = sorted(self.documents, key=lambda x: x.access_count, reverse=True)[:5]
            stats.top_accessed_docs = [
                {'id': doc.id, 'count': doc.access_count} for doc in top_docs
            ]
        
        return stats
    
    def optimize(self):
        """Оптимизация хранилища для производительности"""
        if not self.documents:
            return
        
        logger.info("Оптимизация векторного хранилища...")
        
        # Сортировка документов по частоте доступа
        # Часто используемые документы будут в начале для лучшей локальности кэша
        sorted_indices = sorted(
            range(len(self.documents)),
            key=lambda i: self.documents[i].access_count,
            reverse=True
        )
        
        # Переупорядочивание
        new_documents = [self.documents[i] for i in sorted_indices]
        new_vectors = self.vectors[sorted_indices]
        
        # Обновление индексов
        new_id_to_index = {}
        for new_idx, old_idx in enumerate(sorted_indices):
            doc_id = self.documents[old_idx].id
            new_id_to_index[doc_id] = new_idx
        
        # Применение изменений
        self.documents = new_documents
        self.vectors = new_vectors
        self.id_to_index = new_id_to_index
        
        # Перестройка индексов метаданных
        self.metadata_index.clear()
        self.text_index.clear()
        
        for idx, doc in enumerate(self.documents):
            self._index_document(doc, idx)
        
        # Очистка кэша
        self.search_cache.clear()
        
        logger.info("Оптимизация завершена")
    
    def save(self, filepath: str, compress: bool = True):
        """
        Сохранение хранилища с опциональным сжатием
        
        Args:
            filepath: Путь к файлу
            compress: Использовать сжатие
        """
        data = {
            'dimension': self.dimension,
            'metric': self.metric.value,
            'normalize': self.normalize,
            'vectors': self.vectors,
            'documents': self.documents,
            'id_to_index': self.id_to_index,
            'metadata_index': dict(self.metadata_index),
            'text_index': dict(self.text_index),
            'analytics': self.analytics if self.enable_analytics else None
        }
        
        if compress:
            with gzip.open(filepath + '.gz', 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Хранилище сохранено (сжато): {filepath}.gz")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Хранилище сохранено: {filepath}")
    
    def load(self, filepath: str):
        """Загрузка хранилища с автоопределением сжатия"""
        # Проверяем сжатый файл
        if Path(filepath + '.gz').exists():
            filepath = filepath + '.gz'
        
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            self.dimension = data['dimension']
            self.metric = SimilarityMetric(data['metric'])
            self.normalize = data['normalize']
            self.vectors = data['vectors']
            self.documents = data['documents']
            self.id_to_index = data['id_to_index']
            self.metadata_index = defaultdict(lambda: defaultdict(list), data['metadata_index'])
            self.text_index = defaultdict(list, data.get('text_index', {}))
            
            if self.enable_analytics and data.get('analytics'):
                self.analytics = data['analytics']
            
            logger.info(f"Хранилище загружено: {len(self.documents)} документов")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки хранилища: {e}")
            raise
