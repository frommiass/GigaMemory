# modules/optimization/module.py

from typing import Dict, Any, Optional, List
from ...core.interfaces import IOptimizer, ProcessingResult

from .cache_manager import CacheManager
from .batch_processor import BatchProcessor


class OptimizationModule(IOptimizer):
    """Модуль оптимизации (кэширование и батчинг)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Инициализируем кэш
        self.cache = CacheManager(
            max_size=config.get('cache_size', 10000),
            max_memory_mb=config.get('cache_memory', 1024),
            eviction_strategy=config.get('eviction', 'lru')
        )
        
        # Батч процессор
        self.batch = BatchProcessor(
            batch_size=config.get('batch_size', 32),
            max_wait_time=config.get('max_wait', 1.0)
        )
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша"""
        return self.cache.get(key)
    
    def cache_put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Добавляет значение в кэш"""
        self.cache.put(key, value, ttl=ttl)
    
    def batch_process(self, tasks: List[Dict], processor_func: callable) -> ProcessingResult:
        """Батчевая обработка задач"""
        try:
            results = []
            
            # Разбиваем на батчи
            batch_size = self.config.get('batch_size', 32)
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                
                # Обрабатываем батч
                batch_results = [processor_func(task) for task in batch]
                results.extend(batch_results)
            
            return ProcessingResult(
                success=True,
                data=results,
                metadata={
                    'total_tasks': len(tasks),
                    'batch_size': batch_size,
                    'batches_processed': (len(tasks) + batch_size - 1) // batch_size
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )