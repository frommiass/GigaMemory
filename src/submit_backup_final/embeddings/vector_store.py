"""
Векторное хранилище для быстрого поиска по эмбеддингам
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import pickle
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Документ в векторном хранилище"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None


@dataclass 
class SearchResult:
    """Результат поиска"""
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None
    rank: int = 0


class VectorStore:
    """Базовое векторное хранилище с numpy"""
    
    def __init__(self, dimension: Optional[int] = None, 
                 metric: str = "cosine",
                 normalize: bool = True):
        """
        Инициализация векторного хранилища
        
        Args:
            dimension: Размерность векторов (определяется автоматически)
            metric: Метрика расстояния (cosine, euclidean, dot)
            normalize: Нормализовать векторы для косинусного сходства
        """
        self.dimension = dimension
        self.metric = metric
        self.normalize = normalize and (metric == "cosine")
        
        # Хранилище
        self.vectors: Optional[np.ndarray] = None
        self.documents: List[VectorDocument] = []
        self.id_to_index: Dict[str, int] = {}
        
        # Индексы для фильтрации
        self.metadata_index: Dict[str, Dict[Any, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        logger.info(f"Создано векторное хранилище: metric={metric}, normalize={normalize}")
    
    def add(self, doc_id: str, vector: np.ndarray, 
            metadata: Optional[Dict] = None,
            text: Optional[str] = None):
        """
        Добавляет один документ в хранилище
        
        Args:
            doc_id: Уникальный идентификатор документа
            vector: Вектор эмбеддинга
            metadata: Метаданные документа
            text: Исходный текст (опционально)
        """
        # Проверяем размерность
        if self.dimension is None:
            self.dimension = vector.shape[0]
        elif vector.shape[0] != self.dimension:
            raise ValueError(f"Неверная размерность вектора: {vector.shape[0]} != {self.dimension}")
        
        # Нормализуем если нужно
        if self.normalize:
            vector = vector / np.linalg.norm(vector)
        
        # Добавляем документ
        doc = VectorDocument(
            id=doc_id,
            vector=vector,
            metadata=metadata or {},
            text=text
        )
        
        # Получаем индекс
        index = len(self.documents)
        self.documents.append(doc)
        self.id_to_index[doc_id] = index
        
        # Обновляем векторную матрицу
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        # Индексируем метаданные
        for key, value in doc.metadata.items():
            self.metadata_index[key][value].append(index)
        
        logger.debug(f"Добавлен документ {doc_id}, всего документов: {len(self.documents)}")
    
    def add_batch(self, documents: List[Tuple[str, np.ndarray, Optional[Dict], Optional[str]]]):
        """
        Добавляет батч документов
        
        Args:
            documents: Список кортежей (doc_id, vector, metadata, text)
        """
        for doc_id, vector, metadata, text in documents:
            self.add(doc_id, vector, metadata, text)
    
    def search(self, query_vector: np.ndarray, 
              k: int = 5,
              filter_metadata: Optional[Dict] = None,
              threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Поиск k ближайших соседей
        
        Args:
            query_vector: Вектор запроса
            k: Количество результатов
            filter_metadata: Фильтр по метаданным
            threshold: Минимальный порог сходства
            
        Returns:
            Список результатов поиска
        """
        if not self.documents:
            return []
        
        # Нормализуем запрос если нужно
        if self.normalize:
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Фильтруем по метаданным если нужно
        if filter_metadata:
            valid_indices = self._get_filtered_indices(filter_metadata)
            if not valid_indices:
                return []
        else:
            valid_indices = list(range(len(self.documents)))
        
        # Вычисляем сходство
        if self.metric == "cosine" or self.metric == "dot":
            # Косинусное сходство через dot product (векторы уже нормализованы)
            scores = np.dot(self.vectors[valid_indices], query_vector)
        elif self.metric == "euclidean":
            # Евклидово расстояние (меньше - лучше)
            distances = np.linalg.norm(self.vectors[valid_indices] - query_vector, axis=1)
            scores = -distances  # Инвертируем для единообразия
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}")
        
        # Применяем порог если нужно
        if threshold is not None:
            mask = scores >= threshold
            valid_indices = [idx for idx, m in zip(valid_indices, mask) if m]
            scores = scores[mask]
        
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
        
        return results
    
    def _get_filtered_indices(self, filter_metadata: Dict) -> List[int]:
        """Получает индексы документов, соответствующих фильтру"""
        valid_indices = None
        
        for key, value in filter_metadata.items():
            if key in self.metadata_index:
                indices = set(self.metadata_index[key].get(value, []))
                
                if valid_indices is None:
                    valid_indices = indices
                else:
                    valid_indices = valid_indices.intersection(indices)
        
        return list(valid_indices) if valid_indices else []
    
    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Получает документ по ID"""
        index = self.id_to_index.get(doc_id)
        if index is not None:
            return self.documents[index]
        return None
    
    def delete(self, doc_id: str) -> bool:
        """
        Удаляет документ из хранилища
        
        Args:
            doc_id: ID документа для удаления
            
        Returns:
            True если документ был удален
        """
        if doc_id not in self.id_to_index:
            return False
        
        # Получаем индекс
        index = self.id_to_index[doc_id]
        
        # Удаляем из векторов
        self.vectors = np.delete(self.vectors, index, axis=0)
        
        # Удаляем документ
        doc = self.documents.pop(index)
        
        # Обновляем индексы
        del self.id_to_index[doc_id]
        for doc_id, idx in self.id_to_index.items():
            if idx > index:
                self.id_to_index[doc_id] = idx - 1
        
        # Обновляем метаданные индекс
        for key, value in doc.metadata.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                indices = self.metadata_index[key][value]
                indices.remove(index)
                # Сдвигаем индексы
                self.metadata_index[key][value] = [
                    idx - 1 if idx > index else idx for idx in indices
                ]
        
        logger.debug(f"Удален документ {doc_id}")
        return True
    
    def clear(self):
        """Очищает хранилище"""
        self.vectors = None
        self.documents.clear()
        self.id_to_index.clear()
        self.metadata_index.clear()
        logger.info("Хранилище очищено")
    
    def size(self) -> int:
        """Возвращает количество документов"""
        return len(self.documents)
    
    def save(self, filepath: str):
        """Сохраняет хранилище на диск"""
        data = {
            'dimension': self.dimension,
            'metric': self.metric,
            'normalize': self.normalize,
            'vectors': self.vectors,
            'documents': self.documents,
            'id_to_index': self.id_to_index,
            'metadata_index': dict(self.metadata_index)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Хранилище сохранено: {self.size()} документов в {filepath}")
    
    def load(self, filepath: str):
        """Загружает хранилище с диска"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.metric = data['metric']
        self.normalize = data['normalize']
        self.vectors = data['vectors']
        self.documents = data['documents']
        self.id_to_index = data['id_to_index']
        self.metadata_index = defaultdict(lambda: defaultdict(list), data['metadata_index'])
        
        logger.info(f"Хранилище загружено: {self.size()} документов из {filepath}")


class SimpleVectorStore(VectorStore):
    """Простое векторное хранилище для быстрого прототипирования"""
    
    def __init__(self):
        super().__init__(metric="cosine", normalize=True)
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, 
                 metadata: Optional[List[Dict]] = None):
        """
        Добавляет тексты с их эмбеддингами
        
        Args:
            texts: Список текстов
            embeddings: Массив эмбеддингов
            metadata: Список метаданных для каждого текста
        """
        metadata = metadata or [{} for _ in texts]
        
        documents = [
            (f"doc_{i}", embedding, meta, text)
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata))
        ]
        
        self.add_batch(documents)
    
    def search_by_text(self, query_embedding: np.ndarray, 
                      k: int = 5) -> List[Tuple[str, float]]:
        """
        Простой поиск, возвращающий тексты и scores
        
        Args:
            query_embedding: Эмбеддинг запроса
            k: Количество результатов
            
        Returns:
            Список кортежей (text, score)
        """
        results = self.search(query_embedding, k)
        return [(r.text, r.score) for r in results if r.text]


class OptimizedVectorStore(VectorStore):
    """Оптимизированное векторное хранилище с индексацией"""
    
    def __init__(self, dimension: Optional[int] = None,
                 metric: str = "cosine",
                 normalize: bool = True,
                 use_faiss: bool = False):
        """
        Инициализация оптимизированного хранилища
        
        Args:
            dimension: Размерность векторов
            metric: Метрика расстояния
            normalize: Нормализовать векторы
            use_faiss: Использовать FAISS если доступен (будет реализовано позже)
        """
        super().__init__(dimension, metric, normalize)
        self.use_faiss = use_faiss
        
        if use_faiss:
            logger.warning("FAISS не реализован, используется numpy backend")
    
    def build_index(self):
        """Строит оптимизированный индекс для быстрого поиска"""
        if self.vectors is not None and len(self.vectors) > 1000:
            logger.info(f"Строим индекс для {len(self.vectors)} векторов")
            # Здесь можно добавить построение KD-tree или другого индекса
            pass
    
    def search_multi(self, query_vectors: np.ndarray, 
                    k: int = 5) -> List[List[SearchResult]]:
        """
        Батчевый поиск для нескольких запросов
        
        Args:
            query_vectors: Массив векторов запросов shape (n_queries, dimension)
            k: Количество результатов для каждого запроса
            
        Returns:
            Список результатов для каждого запроса
        """
        results = []
        for query_vector in query_vectors:
            results.append(self.search(query_vector, k))
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику хранилища"""
        stats = {
            'total_documents': self.size(),
            'dimension': self.dimension,
            'metric': self.metric,
            'memory_usage_mb': 0,
            'metadata_keys': list(self.metadata_index.keys())
        }
        
        if self.vectors is not None:
            stats['memory_usage_mb'] = self.vectors.nbytes / (1024 * 1024)
        
        return stats