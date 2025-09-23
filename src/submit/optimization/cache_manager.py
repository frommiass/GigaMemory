"""
Менеджер кэширования для оптимизации производительности
"""
import pickle
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Запись в кэше"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    ttl: Optional[int] = None  # Time to live в секундах
    
    def is_expired(self) -> bool:
        """Проверяет, истек ли срок записи"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Обновляет время последнего доступа"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheManager:
    """Универсальный менеджер кэширования с поддержкой разных стратегий"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 1024,
                 eviction_strategy: str = "lru", ttl_seconds: Optional[int] = None):
        """
        Инициализация менеджера кэша
        
        Args:
            max_size: Максимальное количество записей
            max_memory_mb: Максимальный размер в мегабайтах
            eviction_strategy: Стратегия вытеснения (lru, lfu, fifo)
            ttl_seconds: Время жизни записей по умолчанию
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_strategy = eviction_strategy
        self.default_ttl = ttl_seconds
        
        # Основное хранилище
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Индексы для быстрого доступа
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.fact_cache: Dict[str, List] = {}
        self.compression_cache: Dict[str, str] = {}
        self.query_cache: Dict[str, str] = {}
        
        # Статистика
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0
        }
        
        # Блокировка для потокобезопасности
        self.lock = threading.RLock()
        
        logger.info(f"CacheManager инициализирован: max_size={max_size}, "
                   f"max_memory={max_memory_mb}MB, strategy={eviction_strategy}")
    
    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """
        Получает значение из кэша
        
        Args:
            key: Ключ
            cache_type: Тип кэша (general, embedding, fact, compression, query)
            
        Returns:
            Значение или None
        """
        with self.lock:
            # Выбираем нужный кэш
            if cache_type == "embedding":
                if key in self.embedding_cache:
                    self.stats['hits'] += 1
                    return self.embedding_cache[key]
            elif cache_type == "fact":
                if key in self.fact_cache:
                    self.stats['hits'] += 1
                    return self.fact_cache[key]
            elif cache_type == "compression":
                if key in self.compression_cache:
                    self.stats['hits'] += 1
                    return self.compression_cache[key]
            elif cache_type == "query":
                if key in self.query_cache:
                    self.stats['hits'] += 1
                    return self.query_cache[key]
            else:
                # Общий кэш
                if key in self.cache:
                    entry = self.cache[key]
                    
                    # Проверяем TTL
                    if entry.is_expired():
                        self._evict_entry(key)
                        self.stats['misses'] += 1
                        return None
                    
                    # Обновляем статистику доступа
                    entry.touch()
                    
                    # Для LRU перемещаем в конец
                    if self.eviction_strategy == "lru":
                        self.cache.move_to_end(key)
                    
                    self.stats['hits'] += 1
                    return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, cache_type: str = "general",
            ttl: Optional[int] = None, force: bool = False):
        """
        Добавляет значение в кэш
        
        Args:
            key: Ключ
            value: Значение
            cache_type: Тип кэша
            ttl: Время жизни записи
            force: Принудительно добавить, даже если превышен лимит
        """
        with self.lock:
            # Вычисляем размер
            size_bytes = self._estimate_size(value)
            
            # Проверяем лимиты
            if not force:
                # Проверка размера памяти
                if self.stats['total_size_bytes'] + size_bytes > self.max_memory_bytes:
                    self._evict_until_space(size_bytes)
                
                # Проверка количества записей
                if len(self.cache) >= self.max_size:
                    self._evict_one()
            
            # Добавляем в соответствующий кэш
            if cache_type == "embedding":
                self.embedding_cache[key] = value
            elif cache_type == "fact":
                self.fact_cache[key] = value
            elif cache_type == "compression":
                self.compression_cache[key] = value
            elif cache_type == "query":
                self.query_cache[key] = value
            else:
                # Создаем запись для общего кэша
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    ttl=ttl or self.default_ttl
                )
                self.cache[key] = entry
                self.stats['total_size_bytes'] += size_bytes
    
    def get_or_compute(self, key: str, compute_fn, cache_type: str = "general",
                      ttl: Optional[int] = None) -> Any:
        """
        Получает значение из кэша или вычисляет его
        
        Args:
            key: Ключ
            compute_fn: Функция для вычисления значения
            cache_type: Тип кэша
            ttl: Время жизни записи
            
        Returns:
            Значение из кэша или вычисленное
        """
        # Пробуем получить из кэша
        value = self.get(key, cache_type)
        if value is not None:
            return value
        
        # Вычисляем значение
        value = compute_fn()
        
        # Сохраняем в кэш
        self.put(key, value, cache_type, ttl)
        
        return value
    
    def _estimate_size(self, obj: Any) -> int:
        """Оценивает размер объекта в байтах"""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, str):
            return len(obj.encode('utf-8'))
        elif isinstance(obj, (list, dict)):
            return len(pickle.dumps(obj))
        else:
            return 1000  # Дефолтная оценка
    
    def _evict_one(self):
        """Вытесняет одну запись согласно стратегии"""
        if not self.cache:
            return
        
        if self.eviction_strategy == "lru":
            # Удаляем самую старую (первую) запись
            key = next(iter(self.cache))
        elif self.eviction_strategy == "lfu":
            # Удаляем с наименьшим количеством обращений
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        else:  # fifo
            # Удаляем самую старую по времени создания
            key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        
        self._evict_entry(key)
    
    def _evict_until_space(self, needed_bytes: int):
        """Вытесняет записи пока не освободится нужное место"""
        while self.stats['total_size_bytes'] + needed_bytes > self.max_memory_bytes:
            if not self.cache:
                break
            self._evict_one()
    
    def _evict_entry(self, key: str):
        """Удаляет запись из кэша"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats['total_size_bytes'] -= entry.size_bytes
            self.stats['evictions'] += 1
    
    def clear(self, cache_type: Optional[str] = None):
        """
        Очищает кэш
        
        Args:
            cache_type: Тип кэша для очистки (None = все)
        """
        with self.lock:
            if cache_type == "embedding":
                self.embedding_cache.clear()
            elif cache_type == "fact":
                self.fact_cache.clear()
            elif cache_type == "compression":
                self.compression_cache.clear()
            elif cache_type == "query":
                self.query_cache.clear()
            elif cache_type is None:
                # Очищаем все
                self.cache.clear()
                self.embedding_cache.clear()
                self.fact_cache.clear()
                self.compression_cache.clear()
                self.query_cache.clear()
                self.stats['total_size_bytes'] = 0
            else:
                self.cache.clear()
                self.stats['total_size_bytes'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        with self.lock:
            total_entries = (
                len(self.cache) + 
                len(self.embedding_cache) + 
                len(self.fact_cache) +
                len(self.compression_cache) +
                len(self.query_cache)
            )
            
            hit_rate = 0.0
            total_requests = self.stats['hits'] + self.stats['misses']
            if total_requests > 0:
                hit_rate = self.stats['hits'] / total_requests
            
            return {
                'total_entries': total_entries,
                'general_entries': len(self.cache),
                'embedding_entries': len(self.embedding_cache),
                'fact_entries': len(self.fact_cache),
                'compression_entries': len(self.compression_cache),
                'query_entries': len(self.query_cache),
                'memory_usage_mb': self.stats['total_size_bytes'] / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions']
            }
    
    def save(self, filepath: str):
        """Сохраняет кэш на диск"""
        data = {
            'cache': dict(self.cache),
            'embedding_cache': self.embedding_cache,
            'fact_cache': self.fact_cache,
            'compression_cache': self.compression_cache,
            'query_cache': self.query_cache,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Кэш сохранен в {filepath}")
    
    def load(self, filepath: str):
        """Загружает кэш с диска"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.cache = OrderedDict(data['cache'])
        self.embedding_cache = data['embedding_cache']
        self.fact_cache = data['fact_cache']
        self.compression_cache = data['compression_cache']
        self.query_cache = data['query_cache']
        self.stats = data['stats']
        
        logger.info(f"Кэш загружен из {filepath}")


class BatchProcessor:
    """Класс для батчевой обработки данных"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 1.0):
        """
        Инициализация батч-процессора
        
        Args:
            batch_size: Размер батча
            max_wait_time: Максимальное время ожидания батча (сек)
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # Очереди для разных типов данных
        self.embedding_queue: List[Tuple[str, str]] = []  # (id, text)
        self.fact_queue: List[Tuple[str, str, str, str]] = []  # (text, session_id, dialogue_id)
        self.compression_queue: List[Tuple[str, str]] = []  # (id, text)
        
        # Таймеры
        self.last_process_time = datetime.now()
        
        # Блокировка
        self.lock = threading.RLock()
        
        logger.info(f"BatchProcessor инициализирован: batch_size={batch_size}")
    
    def add_embedding_task(self, task_id: str, text: str) -> bool:
        """Добавляет задачу на создание эмбеддинга"""
        with self.lock:
            self.embedding_queue.append((task_id, text))
            
            # Проверяем, нужно ли обработать батч
            if self._should_process_batch(len(self.embedding_queue)):
                return True
            return False
    
    def add_fact_extraction_task(self, text: str, session_id: str, dialogue_id: str) -> bool:
        """Добавляет задачу на извлечение фактов"""
        with self.lock:
            self.fact_queue.append((text, session_id, dialogue_id))
            
            if self._should_process_batch(len(self.fact_queue)):
                return True
            return False
    
    def add_compression_task(self, task_id: str, text: str) -> bool:
        """Добавляет задачу на сжатие текста"""
        with self.lock:
            self.compression_queue.append((task_id, text))
            
            if self._should_process_batch(len(self.compression_queue)):
                return True
            return False
    
    def _should_process_batch(self, queue_size: int) -> bool:
        """Определяет, нужно ли обработать батч"""
        # По размеру
        if queue_size >= self.batch_size:
            return True
        
        # По времени
        time_elapsed = (datetime.now() - self.last_process_time).total_seconds()
        if queue_size > 0 and time_elapsed >= self.max_wait_time:
            return True
        
        return False
    
    def get_embedding_batch(self) -> List[Tuple[str, str]]:
        """Получает батч для создания эмбеддингов"""
        with self.lock:
            batch = self.embedding_queue[:self.batch_size]
            self.embedding_queue = self.embedding_queue[self.batch_size:]
            self.last_process_time = datetime.now()
            return batch
    
    def get_fact_batch(self) -> List[Tuple[str, str, str]]:
        """Получает батч для извлечения фактов"""
        with self.lock:
            batch = self.fact_queue[:self.batch_size]
            self.fact_queue = self.fact_queue[self.batch_size:]
            self.last_process_time = datetime.now()
            return batch
    
    def get_compression_batch(self) -> List[Tuple[str, str]]:
        """Получает батч для сжатия"""
        with self.lock:
            batch = self.compression_queue[:self.batch_size]
            self.compression_queue = self.compression_queue[self.batch_size:]
            self.last_process_time = datetime.now()
            return batch
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Возвращает размеры очередей"""
        with self.lock:
            return {
                'embedding_queue': len(self.embedding_queue),
                'fact_queue': len(self.fact_queue),
                'compression_queue': len(self.compression_queue)
            }
