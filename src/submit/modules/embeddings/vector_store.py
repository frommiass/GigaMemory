# src/submit/modules/embeddings/vector_store.py
"""
Векторное хранилище для эффективного поиска
"""
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import logging
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """Базовое векторное хранилище с поддержкой различных метрик"""
    
    def __init__(self, metric: str = "cosine", index_type: str = "flat"):
        """
        Инициализация векторного хранилища
        
        Args:
            metric: Метрика сходства (cosine, euclidean, dot)
            index_type: Тип индекса (flat, hnsw, annoy)
        """
        self.metric = metric
        self.index_type = index_type
        
        # Хранилища по диалогам
        self.dialogue_vectors = {}  # dialogue_id -> vectors array
        self.dialogue_texts = {}    # dialogue_id -> list of texts
        self.dialogue_metadata = {}  # dialogue_id -> list of metadata
        
        # Статистика
        self.stats = {
            'total_vectors': 0,
            'total_searches': 0,
            'dialogues_count': 0
        }
        
        logger.info(f"VectorStore инициализирован: metric={metric}, index={index_type}")
    
    def add_vectors(self, dialogue_id: str, session_id: str,
                   vectors: np.ndarray, texts: List[str],
                   metadata: Optional[List[Dict]] = None):
        """
        Добавляет векторы в хранилище
        
        Args:
            dialogue_id: ID диалога
            session_id: ID сессии
            vectors: Матрица векторов
            texts: Соответствующие тексты
            metadata: Дополнительные метаданные
        """
        if len(vectors) != len(texts):
            raise ValueError("Количество векторов должно совпадать с количеством текстов")
        
        # Инициализируем хранилище для диалога если нужно
        if dialogue_id not in self.dialogue_vectors:
            self.dialogue_vectors[dialogue_id] = vectors
            self.dialogue_texts[dialogue_id] = texts
            self.dialogue_metadata[dialogue_id] = metadata or [{} for _ in texts]
            self.stats['dialogues_count'] += 1
        else:
            # Добавляем к существующим
            self.dialogue_vectors[dialogue_id] = np.vstack([
                self.dialogue_vectors[dialogue_id],
                vectors
            ])
            self.dialogue_texts[dialogue_id].extend(texts)
            
            if metadata:
                self.dialogue_metadata[dialogue_id].extend(metadata)
            else:
                self.dialogue_metadata[dialogue_id].extend([{} for _ in texts])
        
        # Добавляем session_id в метаданные
        start_idx = len(self.dialogue_texts[dialogue_id]) - len(texts)
        for i in range(len(texts)):
            self.dialogue_metadata[dialogue_id][start_idx + i]['session_id'] = session_id
        
        self.stats['total_vectors'] += len(vectors)
        
        logger.debug(f"Добавлено {len(vectors)} векторов для диалога {dialogue_id}")
    
    def search(self, dialogue_id: str, query_vector: np.ndarray,
              top_k: int = 5, threshold: Optional[float] = None) -> List[Dict]:
        """
        Поиск похожих векторов
        
        Args:
            dialogue_id: ID диалога
            query_vector: Вектор запроса
            top_k: Количество результатов
            threshold: Минимальный порог сходства
            
        Returns:
            Список результатов с текстами и scores
        """
        self.stats['total_searches'] += 1
        
        # Проверяем наличие диалога
        if dialogue_id not in self.dialogue_vectors:
            logger.debug(f"Диалог {dialogue_id} не найден в хранилище")
            return []
        
        vectors = self.dialogue_vectors[dialogue_id]
        texts = self.dialogue_texts[dialogue_id]
        metadata = self.dialogue_metadata[dialogue_id]
        
        if len(vectors) == 0:
            return []
        
        # Вычисляем сходство
        if self.metric == "cosine":
            # Нормализуем векторы для косинусного сходства
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(vectors_norm, query_norm)
        elif self.metric == "euclidean":
            # Евклидово расстояние (инвертированное)
            distances = np.linalg.norm(vectors - query_vector, axis=1)
            scores = 1.0 / (1.0 + distances)
        elif self.metric == "dot":
            # Скалярное произведение
            scores = np.dot(vectors, query_vector)
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}")
        
        # Применяем порог если задан
        if threshold is not None:
            valid_indices = np.where(scores >= threshold)[0]
        else:
            valid_indices = np.arange(len(scores))
        
        if len(valid_indices) == 0:
            return []
        
        # Находим топ-k
        k = min(top_k, len(valid_indices))
        top_indices_local = np.argpartition(scores[valid_indices], -k)[-k:]
        top_indices = valid_indices[top_indices_local]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Формируем результаты
        results = []
        for idx in top_indices:
            result = {
                'text': texts[idx],
                'score': float(scores[idx]),
                'metadata': metadata[idx].copy(),
                'index': int(idx)
            }
            results.append(result)
        
        return results
    
    def get_dialogue_stats(self, dialogue_id: str) -> Dict[str, Any]:
        """Получает статистику по диалогу"""
        if dialogue_id not in self.dialogue_vectors:
            return {'exists': False}
        
        vectors = self.dialogue_vectors[dialogue_id]
        return {
            'exists': True,
            'vectors_count': len(vectors),
            'dimensions': vectors.shape[1] if len(vectors) > 0 else 0,
            'sessions': len(set(m.get('session_id', '') for m in self.dialogue_metadata[dialogue_id]))
        }
    
    def clear_dialogue(self, dialogue_id: str):
        """Очищает данные диалога"""
        if dialogue_id in self.dialogue_vectors:
            count = len(self.dialogue_vectors[dialogue_id])
            del self.dialogue_vectors[dialogue_id]
            del self.dialogue_texts[dialogue_id]
            del self.dialogue_metadata[dialogue_id]
            self.stats['total_vectors'] -= count
            self.stats['dialogues_count'] -= 1
            logger.info(f"Очищены данные диалога {dialogue_id}")
    
    def save(self, dialogue_id: str, filepath: str):
        """Сохраняет индекс диалога на диск"""
        if dialogue_id not in self.dialogue_vectors:
            logger.warning(f"Диалог {dialogue_id} не найден")
            return False
        
        data = {
            'vectors': self.dialogue_vectors[dialogue_id],
            'texts': self.dialogue_texts[dialogue_id],
            'metadata': self.dialogue_metadata[dialogue_id],
            'metric': self.metric
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Индекс диалога {dialogue_id} сохранен в {filepath}")
        return True
    
    def load(self, dialogue_id: str, filepath: str):
        """Загружает индекс диалога с диска"""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Файл {filepath} не найден")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.dialogue_vectors[dialogue_id] = data['vectors']
            self.dialogue_texts[dialogue_id] = data['texts']
            self.dialogue_metadata[dialogue_id] = data['metadata']
            
            # Обновляем статистику
            self.stats['total_vectors'] += len(data['vectors'])
            self.stats['dialogues_count'] += 1
            
            logger.info(f"Индекс диалога {dialogue_id} загружен из {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает общую статистику"""
        return {
            **self.stats,
            'dialogues': list(self.dialogue_vectors.keys()),
            'metric': self.metric,
            'index_type': self.index_type
        }