"""
Улучшенная система векторного поиска для GigaMemory
Версия 2.0 с расширенной функциональностью
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from transformers import AutoTokenizer, AutoModel
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json
from collections import defaultdict, deque
from datetime import datetime
import warnings
from enum import Enum
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


# ============== Улучшенные конфигурации ==============

class PoolingStrategy(Enum):
    """Стратегии пулинга для эмбеддингов"""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    MEAN_MAX = "mean_max"  # Конкатенация mean и max
    WEIGHTED_MEAN = "weighted_mean"  # С весами по важности токенов


class SimilarityMetric(Enum):
    """Метрики сходства"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"
    ANGULAR = "angular"  # Угловое расстояние


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
    
    # Новые параметры
    use_amp: bool = True  # Automatic Mixed Precision для ускорения
    num_workers: int = 4  # Для параллельной обработки
    prefetch_batches: int = 2  # Предзагрузка батчей
    warmup_steps: int = 0  # Прогрев модели
    compile_model: bool = False  # torch.compile для ускорения (PyTorch 2.0+)
    quantize_model: bool = False  # Квантизация для уменьшения памяти
    
    def __post_init__(self):
        """Валидация и коррекция параметров"""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступна, переключаемся на CPU")
            self.device = "cpu"
            self.use_amp = False
        
        if self.compile_model and not hasattr(torch, 'compile'):
            logger.warning("torch.compile недоступен, отключаем компиляцию")
            self.compile_model = False


# ============== Улучшенный EmbeddingEngine ==============

class ImprovedEmbeddingEngine:
    """Улучшенный движок для создания эмбеддингов с оптимизациями"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        
        # Thread-safe кэш
        self.cache = {} if self.config.use_cache else None
        self.cache_lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'total_encoded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'encoding_time': 0.0,
            'last_batch_time': 0.0
        }
        
        # Загрузка модели с обработкой ошибок
        self._load_model()
        
        # AMP scaler для mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        # Thread pool для параллельной обработки
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
        # Прогрев модели
        if self.config.warmup_steps > 0:
            self._warmup_model()
    
    def _load_model(self):
        """Загружает модель с обработкой ошибок и оптимизациями"""
        try:
            logger.info(f"Загружаем модель {self.config.model_name}")
            
            # Загрузка с кэшированием
            cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=cache_dir
            )
            
            # Загружаем модель
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if self.config.use_amp else torch.float32
            )
            
            # Квантизация если нужна
            if self.config.quantize_model:
                self._quantize_model()
            
            # Компиляция для ускорения (PyTorch 2.0+)
            if self.config.compile_model:
                self.model = torch.compile(self.model)
            
            # Переносим на устройство
            self.device = torch.device(self.config.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Отключаем градиенты
            for param in self.model.parameters():
                param.requires_grad = False
            
            logger.info(f"Модель загружена на {self.device}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise RuntimeError(f"Не удалось загрузить модель {self.config.model_name}: {e}")
    
    def _quantize_model(self):
        """Квантизация модели для уменьшения памяти"""
        if hasattr(torch, 'quantization'):
            logger.info("Применяем квантизацию модели")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
    
    def _warmup_model(self):
        """Прогрев модели для стабильной производительности"""
        logger.info(f"Прогрев модели: {self.config.warmup_steps} шагов")
        dummy_texts = ["Прогрев модели"] * min(self.config.batch_size, 8)
        
        for _ in range(self.config.warmup_steps):
            self._encode_batch(dummy_texts)
        
        self.stats['total_encoded'] = 0  # Сбрасываем после прогрева
    
    def encode(self, texts: Union[str, List[str]], 
               show_progress: bool = False,
               return_tensors: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Улучшенное кодирование с дополнительными опциями
        
        Args:
            texts: Текст или список текстов
            show_progress: Показывать прогресс
            return_tensors: Вернуть torch тензоры вместо numpy
        
        Returns:
            Эмбеддинги в виде numpy array или torch tensor
        """
        start_time = time.time()
        
        # Приводим к списку
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Валидация
        texts = self._validate_texts(texts)
        
        # Получаем эмбеддинги с кэшированием
        embeddings = self._encode_with_cache(texts, show_progress)
        
        # Статистика
        encoding_time = time.time() - start_time
        self.stats['encoding_time'] += encoding_time
        self.stats['last_batch_time'] = encoding_time
        self.stats['total_encoded'] += len(texts)
        
        # Конвертация результата
        if not return_tensors and isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        elif return_tensors and isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        # Возвращаем одно значение для одного текста
        if single_text:
            return embeddings[0]
        
        return embeddings
    
    def _validate_texts(self, texts: List[str]) -> List[str]:
        """Валидация и очистка текстов"""
        validated = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # Убираем лишние пробелы
            text = ' '.join(text.split())
            
            # Ограничиваем длину если слишком длинный
            if len(text) > self.config.max_length * 6:  # Примерная оценка
                text = text[:self.config.max_length * 6]
            
            validated.append(text)
        
        return validated
    
    def _encode_with_cache(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Кодирование с использованием кэша"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.config.use_cache:
            with self.cache_lock:
                for i, text in enumerate(texts):
                    text_hash = self._get_text_hash(text)
                    if text_hash in self.cache:
                        embeddings.append(self.cache[text_hash])
                        self.stats['cache_hits'] += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                        self.stats['cache_misses'] += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Кодируем некэшированные
        if uncached_texts:
            new_embeddings = self._encode_batch_parallel(uncached_texts, show_progress)
            
            # Добавляем в кэш
            if self.config.use_cache:
                with self.cache_lock:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        text_hash = self._get_text_hash(text)
                        self.cache[text_hash] = embedding
            
            # Объединяем результаты
            if embeddings:
                result = np.zeros((len(texts), new_embeddings.shape[1]))
                cached_idx = 0
                new_idx = 0
                
                for i in range(len(texts)):
                    if i in uncached_indices:
                        result[i] = new_embeddings[new_idx]
                        new_idx += 1
                    else:
                        result[i] = embeddings[cached_idx]
                        cached_idx += 1
                
                embeddings = result
            else:
                embeddings = new_embeddings
        else:
            embeddings = np.array(embeddings)
        
        return embeddings
    
    def _encode_batch_parallel(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Параллельное кодирование батчей"""
        if len(texts) <= self.config.batch_size:
            # Один батч - не нужна параллелизация
            return self._encode_batch(texts, show_progress)
        
        # Разбиваем на батчи
        batches = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batches.append(batch)
        
        # Параллельная обработка
        futures = []
        for batch in batches:
            future = self.executor.submit(self._encode_batch, batch, False)
            futures.append(future)
        
        # Собираем результаты
        all_embeddings = []
        for i, future in enumerate(as_completed(futures)):
            if show_progress:
                logger.info(f"Обработан батч {i+1}/{len(batches)}")
            embeddings = future.result()
            all_embeddings.append(embeddings)
        
        # Восстанавливаем порядок
        return np.vstack(all_embeddings)
    
    def _encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Оптимизированное кодирование батча"""
        # Токенизация
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Переносим на устройство
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Получаем эмбеддинги с AMP
        with torch.no_grad():
            if self.config.use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)
            
            # Применяем стратегию пулинга
            embeddings = self._apply_pooling(
                outputs.last_hidden_state,
                encoded['attention_mask']
            )
            
            # Нормализация
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Конвертируем в numpy
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def _apply_pooling(self, hidden_states: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Расширенные стратегии пулинга"""
        strategy = self.config.pooling_strategy
        
        if strategy == PoolingStrategy.CLS:
            return hidden_states[:, 0, :]
        
        elif strategy == PoolingStrategy.MAX:
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), -1e9
            )
            return torch.max(hidden_states, dim=1)[0]
        
        elif strategy == PoolingStrategy.MEAN:
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif strategy == PoolingStrategy.MEAN_MAX:
            # Комбинация mean и max pooling
            mean_pool = self._apply_mean_pooling(hidden_states, attention_mask)
            max_pool = self._apply_max_pooling(hidden_states, attention_mask)
            return torch.cat([mean_pool, max_pool], dim=1)
        
        elif strategy == PoolingStrategy.WEIGHTED_MEAN:
            # Взвешенное среднее с убыванием веса
            seq_len = hidden_states.size(1)
            weights = torch.arange(seq_len, 0, -1, dtype=torch.float32, device=hidden_states.device)
            weights = weights.unsqueeze(0).unsqueeze(-1)
            
            weighted_hidden = hidden_states * weights
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            sum_embeddings = torch.sum(weighted_hidden * attention_mask_expanded, 1)
            sum_weights = torch.sum(weights * attention_mask_expanded, 1)
            sum_weights = torch.clamp(sum_weights, min=1e-9)
            
            return sum_embeddings / sum_weights
        
        else:
            return self._apply_mean_pooling(hidden_states, attention_mask)
    
    def _apply_mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling helper"""
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _apply_max_pooling(self, hidden_states, attention_mask):
        """Max pooling helper"""
        hidden_states = hidden_states.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), -1e9
        )
        return torch.max(hidden_states, dim=1)[0]
    
    def _get_text_hash(self, text: str) -> str:
        """Улучшенное хэширование с учетом конфигурации"""
        # Включаем параметры модели в хэш
        config_str = f"{self.config.model_name}_{self.config.pooling_strategy.value}_{self.config.max_length}"
        text_to_hash = f"{config_str}_{text}"
        return hashlib.md5(text_to_hash.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        stats = self.stats.copy()
        if self.config.use_cache:
            stats['cache_size'] = len(self.cache)
            stats['cache_hit_rate'] = (
                self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            )
        return stats
    
    def clear_cache(self):
        """Потокобезопасная очистка кэша"""
        if self.cache:
            with self.cache_lock:
                self.cache.clear()
                self.stats['cache_hits'] = 0
                self.stats['cache_misses'] = 0
            logger.info("Кэш очищен")
    
    def __del__(self):
        """Корректное освобождение ресурсов"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
