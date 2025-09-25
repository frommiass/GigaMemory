# modules/embeddings/improved_vector_store.py
"""
Улучшенное векторное хранилище с поддержкой FAISS
"""

import numpy as np
from typing import List, Dict, Optional, Any
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ImprovedVectorStore:
    """
    Векторное хранилище с FAISS для быстрого поиска
    Fallback на numpy если FAISS недоступен
    """
    
    def __init__(self, use_faiss: bool = True, metric: str = "cosine"):
        self.metric = metric
        self.use_faiss = use_faiss
        
        # Пробуем инициализировать FAISS
        self.faiss_available = False
        self.faiss_indices = {}  # dialogue_id -> faiss index
        
        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
                self.faiss_available = True
                logger.info("FAISS успешно инициализирован")
            except ImportError:
                logger.warning("FAISS не установлен, используем numpy fallback")
                self.faiss_available = False
        
        # Fallback хранилища
        self.numpy_vectors = {}  # dialogue_id -> vectors
        self.texts = {}  # dialogue_id -> texts
        self.metadata = {}  # dialogue_id -> metadata
        
        # Размерность векторов (для rubert-tiny2)
        self.dim = 312
        
        # Статистика
        self.stats = {
            'total_vectors': 0,
            'total_searches': 0,
            'dialogues': 0
        }
    
    def _create_faiss_index(self, dialogue_id: str):
        """Создаёт FAISS индекс для диалога"""
        if not self.faiss_available:
            return None
        
        # Квантизированный индекс для экономии памяти
        # IVF - Inverted File index для кластеризации
        # PQ - Product Quantization для сжатия векторов
        
        # Для маленьких датасетов используем простой индекс
        if dialogue_id not in self.faiss_indices:
            # Определяем количество кластеров
            n_clusters = min(100, self.stats['total_vectors'] // 10 + 1)
            
            if n_clusters < 10:
                # Простой индекс для малых данных
                if self.metric == "cosine":
                    index = self.faiss.IndexFlatIP(self.dim)  # Inner Product для косинусного
                else:
                    index = self.faiss.IndexFlatL2(self.dim)  # L2 для евклидова
            else:
                # Квантизованный индекс для больших данных
                quantizer = self.faiss.IndexFlatL2(self.dim)
                
                # 8 байт на вектор вместо 312*4=1248 байт!
                # Экономия памяти ~150x
                index = self.faiss.IndexIVFPQ(
                    quantizer,
                    self.dim,
                    n_clusters,
                    8,  # bytes per vector
                    8   # bits per sub-quantizer
                )
                
                # Флаг что индекс нужно обучить
                index.is_trained = False
            
            self.faiss_indices[dialogue_id] = index
            logger.info(f"Создан FAISS индекс для {dialogue_id}, кластеров: {n_clusters}")
        
        return self.faiss_indices[dialogue_id]
    
    def add_batch(self, dialogue_id: str, vectors: np.ndarray,
                  texts: List[str], metadata: List[Dict] = None):
        """
        Добавляет батч векторов в хранилище
        
        Args:
            dialogue_id: ID диалога
            vectors: Матрица векторов (N x dim)
            texts: Список текстов
            metadata: Список метаданных
        """
        if len(vectors) != len(texts):
            raise ValueError("Количество векторов должно совпадать с текстами")
        
        # Сохраняем тексты и метаданные
        if dialogue_id not in self.texts:
            self.texts[dialogue_id] = []
            self.metadata[dialogue_id] = []
            self.stats['dialogues'] += 1
        
        self.texts[dialogue_id].extend(texts)
        
        if metadata:
            self.metadata[dialogue_id].extend(metadata)
        else:
            self.metadata[dialogue_id].extend([{}] * len(texts))
        
        # Добавляем векторы
        if self.faiss_available and self.use_faiss:
            # Используем FAISS
            index = self._create_faiss_index(dialogue_id)
            
            # Нормализуем для косинусного сходства
            if self.metric == "cosine":
                # Нормализация для Inner Product = Cosine Similarity
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / (norms + 1e-8)
            
            # Обучаем индекс если нужно
            if hasattr(index, 'is_trained') and not index.is_trained:
                if len(vectors) >= 100:
                    # Обучаем на текущем батче
                    index.train(vectors)
                    index.is_trained = True
                    logger.info(f"FAISS индекс обучен для {dialogue_id}")
                else:
                    # Накапливаем векторы для обучения
                    if dialogue_id not in self.numpy_vectors:
                        self.numpy_vectors[dialogue_id] = []
                    self.numpy_vectors[dialogue_id].append(vectors)
                    
                    # Проверяем, достаточно ли векторов
                    total_vecs = sum(len(v) for v in self.numpy_vectors[dialogue_id])
                    if total_vecs >= 100:
                        # Обучаем на всех накопленных
                        all_vecs = np.vstack(self.numpy_vectors[dialogue_id])
                        index.train(all_vecs)
                        index.is_trained = True
                        index.add(all_vecs)
                        # Очищаем временное хранилище
                        del self.numpy_vectors[dialogue_id]
                        logger.info(f"FAISS индекс обучен на {total_vecs} векторах")
                    
                    self.stats['total_vectors'] += len(vectors)
                    return  # Не добавляем пока не обучен
            
            # Добавляем векторы в индекс
            index.add(vectors)
            
        else:
            # Fallback на numpy
            if dialogue_id not in self.numpy_vectors:
                self.numpy_vectors[dialogue_id] = vectors
            else:
                self.numpy_vectors[dialogue_id] = np.vstack([
                    self.numpy_vectors[dialogue_id],
                    vectors
                ])
        
        self.stats['total_vectors'] += len(vectors)
        logger.debug(f"Добавлено {len(vectors)} векторов для {dialogue_id}")
    
    def search(self, dialogue_id: str, query_vector: np.ndarray,
              top_k: int = 5, threshold: float = None) -> List[Dict]:
        """
        Поиск похожих векторов
        
        Args:
            dialogue_id: ID диалога
            query_vector: Вектор запроса (1D array)
            top_k: Количество результатов
            threshold: Минимальный порог сходства
        
        Returns:
            Список результатов с текстами и scores
        """
        self.stats['total_searches'] += 1
        
        if dialogue_id not in self.texts:
            logger.debug(f"Диалог {dialogue_id} не найден")
            return []
        
        texts = self.texts[dialogue_id]
        metadata = self.metadata[dialogue_id]
        
        if not texts:
            return []
        
        # Поиск через FAISS
        if self.faiss_available and dialogue_id in self.faiss_indices:
            index = self.faiss_indices[dialogue_id]
            
            # Проверяем что индекс не пустой
            if index.ntotal == 0:
                return []
            
            # Нормализуем запрос для косинусного сходства
            if self.metric == "cosine":
                query_norm = np.linalg.norm(query_vector)
                query_vector = query_vector / (query_norm + 1e-8)
            
            # Reshape для FAISS (нужна 2D матрица)
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            
            # Поиск
            if hasattr(index, 'nprobe'):
                # Для IVF индексов увеличиваем точность поиска
                index.nprobe = min(10, index.nlist)
            
            distances, indices = index.search(query_vector, min(top_k, index.ntotal))
            
            # Формируем результаты
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(texts):  # Валидный индекс
                    # Конвертируем расстояние в сходство
                    if self.metric == "cosine":
                        score = float(dist)  # Inner product уже даёт сходство
                    else:
                        score = 1.0 / (1.0 + float(dist))  # Инвертируем расстояние
                    
                    # Применяем порог
                    if threshold and score < threshold:
                        continue
                    
                    results.append({
                        'text': texts[idx],
                        'score': score,
                        'metadata': metadata[idx],
                        'index': int(idx)
                    })
            
            return results
        
        # Fallback на numpy поиск
        elif dialogue_id in self.numpy_vectors:
            vectors = self.numpy_vectors[dialogue_id]
            
            # Вычисляем сходство
            if self.metric == "cosine":
                # Косинусное сходство
                query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
                vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
                scores = np.dot(vectors_norm, query_norm)
            else:
                # Евклидово расстояние
                distances = np.linalg.norm(vectors - query_vector, axis=1)
                scores = 1.0 / (1.0 + distances)
            
            # Применяем порог
            if threshold:
                valid_idx = np.where(scores >= threshold)[0]
            else:
                valid_idx = np.arange(len(scores))
            
            if len(valid_idx) == 0:
                return []
            
            # Топ-k
            k = min(top_k, len(valid_idx))
            top_indices = np.argpartition(scores[valid_idx], -k)[-k:]
            top_indices = valid_idx[top_indices]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            
            # Результаты
            results = []
            for idx in top_indices:
                results.append({
                    'text': texts[idx],
                    'score': float(scores[idx]),
                    'metadata': metadata[idx],
                    'index': int(idx)
                })
            
            return results
        
        return []
    
    def clear_dialogue(self, dialogue_id: str):
        """Очищает данные диалога"""
        if dialogue_id in self.texts:
            del self.texts[dialogue_id]
            del self.metadata[dialogue_id]
        
        if dialogue_id in self.numpy_vectors:
            del self.numpy_vectors[dialogue_id]
        
        if dialogue_id in self.faiss_indices:
            del self.faiss_indices[dialogue_id]
        
        self.stats['dialogues'] -= 1
        logger.info(f"Очищены данные диалога {dialogue_id}")
    
    def save(self, dialogue_id: str, filepath: str) -> bool:
        """Сохраняет индекс на диск"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'texts': self.texts.get(dialogue_id, []),
                'metadata': self.metadata.get(dialogue_id, []),
                'metric': self.metric
            }
            
            # Сохраняем основные данные
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Сохраняем FAISS индекс отдельно
            if self.faiss_available and dialogue_id in self.faiss_indices:
                index_path = filepath.with_suffix('.faiss')
                self.faiss.write_index(self.faiss_indices[dialogue_id], str(index_path))
                logger.info(f"FAISS индекс сохранён: {index_path}")
            
            # Или numpy векторы
            elif dialogue_id in self.numpy_vectors:
                vectors_path = filepath.with_suffix('.npy')
                np.save(vectors_path, self.numpy_vectors[dialogue_id])
                logger.info(f"Векторы сохранены: {vectors_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            return False
    
    def load(self, dialogue_id: str, filepath: str) -> bool:
        """Загружает индекс с диска"""
        try:
            filepath = Path(filepath)
            
            # Загружаем основные данные
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.texts[dialogue_id] = data['texts']
            self.metadata[dialogue_id] = data['metadata']
            
            # Пробуем загрузить FAISS индекс
            if self.faiss_available:
                index_path = filepath.with_suffix('.faiss')
                if index_path.exists():
                    self.faiss_indices[dialogue_id] = self.faiss.read_index(str(index_path))
                    logger.info(f"FAISS индекс загружен: {index_path}")
                    return True
            
            # Или numpy векторы
            vectors_path = filepath.with_suffix('.npy')
            if vectors_path.exists():
                self.numpy_vectors[dialogue_id] = np.load(vectors_path)
                logger.info(f"Векторы загружены: {vectors_path}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        return {
            **self.stats,
            'backend': 'FAISS' if self.faiss_available else 'NumPy',
            'indices': list(self.faiss_indices.keys()) if self.faiss_available else [],
            'metric': self.metric
        }