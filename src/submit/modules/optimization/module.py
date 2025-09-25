# modules/optimization/module.py

import os
import time
import pickle
import zlib
import hashlib
import logging
import threading
import psutil
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import queue

from ...core.interfaces import IOptimizer, ProcessingResult
from .cache_manager import CacheManager, CacheEntry
from .batch_processor import BatchProcessor, EmbeddingBatchProcessor, FactExtractionBatchProcessor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    cache_hits: int = 0
    cache_misses: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    batch_processed: int = 0
    total_batch_time: float = 0.0
    batch_count: int = 0
    memory_usage_bytes: int = 0
    cpu_usage_percent: float = 0.0
    
    @property
    def avg_batch_time(self) -> float:
        """Среднее время обработки батча"""
        return self.total_batch_time / self.batch_count if self.batch_count > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Процент попаданий в кэш"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceMonitor:
    """Монитор производительности системы"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.lock = threading.RLock()
        self.process = psutil.Process()
        
        # История метрик для анализа трендов
        self.history = []
        self.max_history_size = 1000
        
        # Запускаем мониторинг ресурсов
        self._start_resource_monitoring()
    
    def _start_resource_monitoring(self):
        """Запускает фоновый мониторинг ресурсов"""
        def monitor_loop():
            while True:
                try:
                    # Обновляем метрики CPU и памяти
                    with self.lock:
                        self.metrics.cpu_usage_percent = self.process.cpu_percent()
                        self.metrics.memory_usage_bytes = self.process.memory_info().rss
                except Exception as e:
                    logger.error(f"Ошибка мониторинга ресурсов: {e}")
                
                time.sleep(5)  # Обновляем каждые 5 секунд
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def record_cache_access(self, hit: bool, level: Optional[str] = None):
        """Записывает обращение к кэшу"""
        with self.lock:
            if hit:
                self.metrics.cache_hits += 1
                if level == 'L1':
                    self.metrics.l1_hits += 1
                elif level == 'L2':
                    self.metrics.l2_hits += 1
                elif level == 'L3':
                    self.metrics.l3_hits += 1
            else:
                self.metrics.cache_misses += 1
    
    def record_batch(self, batch_size: int, processing_time: float):
        """Записывает обработку батча"""
        with self.lock:
            self.metrics.batch_processed += batch_size
            self.metrics.total_batch_time += processing_time
            self.metrics.batch_count += 1
            
            # Сохраняем в историю
            self._add_to_history({
                'timestamp': datetime.now(),
                'batch_size': batch_size,
                'processing_time': processing_time
            })
    
    def _add_to_history(self, record: Dict):
        """Добавляет запись в историю"""
        self.history.append(record)
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
    
    def get_report(self) -> Dict[str, Any]:
        """Возвращает отчет о производительности"""
        with self.lock:
            return {
                'cache_hit_rate': self.metrics.cache_hit_rate,
                'cache_stats': {
                    'hits': self.metrics.cache_hits,
                    'misses': self.metrics.cache_misses,
                    'l1_hits': self.metrics.l1_hits,
                    'l2_hits': self.metrics.l2_hits,
                    'l3_hits': self.metrics.l3_hits
                },
                'batch_stats': {
                    'total_processed': self.metrics.batch_processed,
                    'batch_count': self.metrics.batch_count,
                    'avg_batch_time_ms': self.metrics.avg_batch_time * 1000
                },
                'resource_usage': {
                    'memory_usage_mb': self.metrics.memory_usage_bytes / 1024 / 1024,
                    'cpu_usage_percent': self.metrics.cpu_usage_percent
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trends(self) -> Dict[str, Any]:
        """Анализирует тренды производительности"""
        if not self.history:
            return {}
        
        # Анализируем последние 100 записей
        recent = self.history[-100:]
        
        # Вычисляем тренды
        processing_times = [r['processing_time'] for r in recent]
        batch_sizes = [r['batch_size'] for r in recent]
        
        return {
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'max_processing_time': max(processing_times),
            'min_processing_time': min(processing_times),
            'avg_batch_size': sum(batch_sizes) / len(batch_sizes),
            'trend': self._calculate_trend(processing_times)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Вычисляет тренд (улучшение/ухудшение)"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Сравниваем среднее первой и последней четверти
        quarter = len(values) // 4
        first_quarter = sum(values[:quarter]) / quarter
        last_quarter = sum(values[-quarter:]) / len(values[-quarter:])
        
        diff_percent = (last_quarter - first_quarter) / first_quarter * 100
        
        if diff_percent < -10:
            return 'improving'
        elif diff_percent > 10:
            return 'degrading'
        else:
            return 'stable'


class TypeHandler:
    """Обработчик для специфичных типов данных"""
    
    def __init__(self):
        self.handlers = {}
    
    def register(self, type_name: str, compress_func: Callable, decompress_func: Callable):
        """Регистрирует обработчик типа"""
        self.handlers[type_name] = {
            'compress': compress_func,
            'decompress': decompress_func
        }
    
    def compress(self, type_name: str, data: Any) -> Any:
        """Сжимает данные"""
        if type_name in self.handlers:
            return self.handlers[type_name]['compress'](data)
        return data
    
    def decompress(self, type_name: str, data: Any) -> Any:
        """Распаковывает данные"""
        if type_name in self.handlers:
            return self.handlers[type_name]['decompress'](data)
        return data


class OptimizationModule(IOptimizer):
    """Расширенный модуль оптимизации с многоуровневым кэшированием"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Многоуровневое кэширование
        self.l1_cache = {}  # In-memory быстрый кэш
        self.l1_max_size = config.get('l1_cache_size', 100)
        
        # L2 - основной кэш
        self.l2_cache = CacheManager(
            max_size=config.get('l2_cache_size', 10000),
            max_memory_mb=config.get('l2_cache_memory', 1024),
            eviction_strategy=config.get('eviction_strategy', 'lru'),
            ttl_seconds=config.get('default_ttl', 3600)
        )
        
        # L3 - дисковый кэш
        self.l3_cache_path = config.get('disk_cache_path')
        if self.l3_cache_path:
            Path(self.l3_cache_path).mkdir(parents=True, exist_ok=True)
        
        # Батч процессор с приоритетами
        self.batch_processor = BatchProcessor(
            batch_size=config.get('batch_size', 32),
            max_wait_time=config.get('max_wait', 1.0),
            num_workers=config.get('num_workers', 4)
        )
        
        # Монитор производительности
        self.monitor = PerformanceMonitor()
        
        # Обработчики типов
        self.type_handler = TypeHandler()
        self._setup_type_handlers()
        
        # Блокировка для потокобезопасности
        self.lock = threading.RLock()
        
        # Настраиваем автоочистку
        self.setup_auto_cleanup()
        
        logger.info(f"OptimizationModule инициализирован с 3-уровневым кэшем")
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Получает значение из многоуровневого кэша"""
        with self.lock:
            # Проверяем L1
            if key in self.l1_cache:
                self.monitor.record_cache_access(True, 'L1')
                self._update_l1_lru(key)
                return self.l1_cache[key]
            
            # Проверяем L2
            value = self.l2_cache.get(key)
            if value is not None:
                self.monitor.record_cache_access(True, 'L2')
                # Поднимаем в L1
                self._promote_to_l1(key, value)
                return value
            
            # Проверяем L3 (диск)
            if self.l3_cache_path:
                value = self._get_from_disk_cache(key)
                if value is not None:
                    self.monitor.record_cache_access(True, 'L3')
                    # Поднимаем в L2 и L1
                    self.l2_cache.put(key, value)
                    self._promote_to_l1(key, value)
                    return value
            
            self.monitor.record_cache_access(False)
            return None
    
    def cache_put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Добавляет значение во все уровни кэша"""
        with self.lock:
            # Добавляем в L1
            self._promote_to_l1(key, value)
            
            # Добавляем в L2
            self.l2_cache.put(key, value, ttl=ttl)
            
            # Добавляем в L3 (асинхронно)
            if self.l3_cache_path:
                threading.Thread(
                    target=self._save_to_disk_cache,
                    args=(key, value),
                    daemon=True
                ).start()
    
    def batch_process(self, tasks: List[Dict], processor_func: callable) -> ProcessingResult:
        """Батчевая обработка с поддержкой приоритетов"""
        try:
            # Регистрируем процессор если нужно
            task_type = 'batch_process'
            if task_type not in self.batch_processor.task_queues:
                self.batch_processor.register_task_type(task_type, processor_func)
            
            # Добавляем задачи
            task_ids = []
            for task in tasks:
                task_id = task.get('id', hashlib.md5(str(task).encode()).hexdigest())
                priority = task.get('priority', 0)
                
                success = self.batch_processor.add_task(
                    task_id=task_id,
                    data=task,
                    task_type=task_type,
                    priority=priority
                )
                
                if success:
                    task_ids.append(task_id)
            
            # Ждем результаты
            results = []
            start_time = time.time()
            
            for _ in range(len(task_ids)):
                result = self.batch_processor.get_result(task_type, timeout=10.0)
                if result:
                    results.append(result)
            
            processing_time = time.time() - start_time
            self.monitor.record_batch(len(tasks), processing_time)
            
            return ProcessingResult(
                success=True,
                data=results,
                metadata={
                    'total_tasks': len(tasks),
                    'processed_tasks': len(results),
                    'processing_time': processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка батчевой обработки: {e}")
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )
    
    def batch_process_priority(self, tasks: List[Dict], processor_func: callable) -> ProcessingResult:
        """Обрабатывает задачи с учетом приоритета"""
        try:
            start_time = time.time()
            
            # Сортируем по приоритету
            priority_tasks = sorted(tasks, key=lambda x: x.get('priority', 0), reverse=True)
            
            # Разделяем на высокоприоритетные и обычные
            high_priority = [t for t in priority_tasks if t.get('priority', 0) >= 8]
            normal_priority = [t for t in priority_tasks if t.get('priority', 0) < 8]
            
            results = []
            
            # Высокоприоритетные - по одной
            for task in high_priority:
                result = processor_func(task)
                results.append(result)
            
            # Обычные - батчами
            batch_size = self.config.get('batch_size', 32)
            for i in range(0, len(normal_priority), batch_size):
                batch = normal_priority[i:i+batch_size]
                batch_results = [processor_func(task) for task in batch]
                results.extend(batch_results)
            
            processing_time = time.time() - start_time
            self.monitor.record_batch(len(tasks), processing_time)
            
            return ProcessingResult(
                success=True,
                data=results,
                metadata={
                    'high_priority_count': len(high_priority),
                    'normal_priority_count': len(normal_priority),
                    'processing_time': processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка приоритетной обработки: {e}")
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )
    
    def get_performance_report(self) -> ProcessingResult:
        """Возвращает отчет о производительности"""
        report = self.monitor.get_report()
        
        # Добавляем статистику кэшей
        report['cache_stats'].update(self.l2_cache.get_stats())
        report['l1_cache_size'] = len(self.l1_cache)
        
        # Добавляем статистику батч-процессора
        report['batch_processor_stats'] = self.batch_processor.get_stats()
        
        # Добавляем тренды
        report['trends'] = self.monitor.get_trends()
        
        return ProcessingResult(
            success=True,
            data=report,
            metadata={'timestamp': datetime.now().isoformat()}
        )
    
    def setup_auto_cleanup(self):
        """Настраивает автоматическую очистку кэша"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Каждый час
                    
                    with self.lock:
                        # Очищаем L1 если слишком большой
                        if len(self.l1_cache) > self.l1_max_size * 2:
                            # Удаляем половину старых записей
                            keys = list(self.l1_cache.keys())[:self.l1_max_size]
                            for key in keys:
                                del self.l1_cache[key]
                            logger.info(f"L1 cache cleaned: {len(keys)} entries removed")
                    
                    # Очищаем просроченные в L2
                    self.l2_cache.clear('expired')
                    
                    # Очищаем старые файлы в L3
                    if self.l3_cache_path:
                        self._cleanup_disk_cache()
                    
                except Exception as e:
                    logger.error(f"Ошибка автоочистки: {e}")
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
        logger.info("Автоочистка кэша запущена")
    
    def _setup_type_handlers(self):
        """Настраивает обработчики для разных типов данных"""
        # Оптимизация для эмбеддингов
        self.type_handler.register(
            'embedding',
            compress_func=lambda x: x.astype(np.float16) if isinstance(x, np.ndarray) else x,
            decompress_func=lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) else x
        )
        
        # Оптимизация для текстов
        self.type_handler.register(
            'text',
            compress_func=lambda x: zlib.compress(x.encode()) if isinstance(x, str) else x,
            decompress_func=lambda x: zlib.decompress(x).decode() if isinstance(x, bytes) else x
        )
        
        # Оптимизация для JSON
        self.type_handler.register(
            'json',
            compress_func=lambda x: zlib.compress(pickle.dumps(x)),
            decompress_func=lambda x: pickle.loads(zlib.decompress(x))
        )
    
    def optimize_for_embeddings(self):
        """Специальная оптимизация для эмбеддингов"""
        self.l2_cache.register_type_handler(
            'embedding',
            compress_func=lambda x: x.astype(np.float16),
            decompress_func=lambda x: x.astype(np.float32)
        )
        logger.info("Оптимизация для эмбеддингов включена")
    
    def optimize_for_text(self):
        """Специальная оптимизация для текстов"""
        self.l2_cache.register_type_handler(
            'text',
            compress_func=lambda x: zlib.compress(x.encode()),
            decompress_func=lambda x: zlib.decompress(x).decode()
        )
        logger.info("Оптимизация для текстов включена")
    
    def _promote_to_l1(self, key: str, value: Any):
        """Поднимает значение в L1 кэш"""
        if len(self.l1_cache) >= self.l1_max_size:
            # Удаляем самый старый элемент (простой FIFO для L1)
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = value
    
    def _update_l1_lru(self, key: str):
        """Обновляет LRU для L1 кэша"""
        # Перемещаем в конец (самый свежий)
        value = self.l1_cache.pop(key)
        self.l1_cache[key] = value
    
    def _get_from_disk_cache(self, key: str) -> Optional[Any]:
        """Получает значение из дискового кэша"""
        filepath = Path(self.l3_cache_path) / f"{key}.pkl"
        
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Ошибка чтения из L3 кэша: {e}")
                # Удаляем поврежденный файл
                filepath.unlink(missing_ok=True)
        
        return None
    
    def _save_to_disk_cache(self, key: str, value: Any):
        """Сохраняет значение в дисковый кэш"""
        if not self.l3_cache_path:
            return
        
        filepath = Path(self.l3_cache_path) / f"{key}.pkl"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Ошибка записи в L3 кэш: {e}")
    
    def _cleanup_disk_cache(self):
        """Очищает старые файлы в дисковом кэше"""
        if not self.l3_cache_path:
            return
        
        cache_dir = Path(self.l3_cache_path)
        max_age_days = self.config.get('l3_cache_max_age_days', 7)
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        removed_count = 0
        for filepath in cache_dir.glob("*.pkl"):
            try:
                # Проверяем возраст файла
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff_time:
                    filepath.unlink()
                    removed_count += 1
            except Exception as e:
                logger.error(f"Ошибка при удалении {filepath}: {e}")
        
        if removed_count > 0:
            logger.info(f"L3 cache cleaned: {removed_count} old files removed")
    
    def clear_all_caches(self):
        """Полная очистка всех уровней кэша"""
        with self.lock:
            # Очищаем L1
            self.l1_cache.clear()
            
            # Очищаем L2
            self.l2_cache.clear()
            
            # Очищаем L3
            if self.l3_cache_path:
                cache_dir = Path(self.l3_cache_path)
                for filepath in cache_dir.glob("*.pkl"):
                    filepath.unlink(missing_ok=True)
        
        logger.info("Все уровни кэша очищены")
    
    def get_cache_sizes(self) -> Dict[str, int]:
        """Возвращает размеры всех кэшей"""
        l3_size = 0
        if self.l3_cache_path:
            cache_dir = Path(self.l3_cache_path)
            l3_size = len(list(cache_dir.glob("*.pkl")))
        
        return {
            'l1_entries': len(self.l1_cache),
            'l2_entries': self.l2_cache.get_stats()['total_entries'],
            'l3_entries': l3_size
        }
    
    def warmup_cache(self, data: List[Tuple[str, Any]]):
        """Прогревает кэш начальными данными"""
        for key, value in data:
            self.cache_put(key, value)
        
        logger.info(f"Cache warmed up with {len(data)} entries")