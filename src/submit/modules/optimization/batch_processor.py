"""
Батчевая обработка данных для оптимизации производительности
"""
import logging
import threading
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Задача для батчевой обработки"""
    task_id: str
    data: Any
    task_type: str
    priority: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def __lt__(self, other):
        """Сравнение для сортировки по приоритету"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Высокий приоритет первым
        return self.created_at < other.created_at  # Старые задачи первыми
    
    def __le__(self, other):
        return self < other or self == other
    
    def __gt__(self, other):
        return not self <= other
    
    def __ge__(self, other):
        return not self < other


class BatchProcessor:
    """Универсальный батч-процессор"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 1.0,
                 num_workers: int = 4, max_queue_size: int = 1000):
        """
        Инициализация батч-процессора
        
        Args:
            batch_size: Размер батча
            max_wait_time: Максимальное время ожидания (сек)
            num_workers: Количество рабочих потоков
            max_queue_size: Максимальный размер очереди
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        
        # Очереди для разных типов задач
        self.task_queues: Dict[str, queue.PriorityQueue] = {}
        self.result_queues: Dict[str, queue.Queue] = {}
        
        # Пул потоков
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Статистика
        self.stats = {
            'total_tasks': 0,
            'processed_tasks': 0,
            'failed_tasks': 0,
            'avg_batch_time': 0.0,
            'queue_sizes': {}
        }
        
        # Блокировка
        self.lock = threading.RLock()
        
        # Фоновые потоки
        self.worker_threads = []
        self.running = False
        
        logger.info(f"BatchProcessor инициализирован: batch_size={batch_size}, "
                   f"workers={num_workers}, max_wait={max_wait_time}s")
    
    def register_task_type(self, task_type: str, processor_func: Callable):
        """
        Регистрирует тип задачи и функцию обработки
        
        Args:
            task_type: Тип задачи
            processor_func: Функция для обработки батча
        """
        self.task_queues[task_type] = queue.PriorityQueue(maxsize=self.max_queue_size)
        self.result_queues[task_type] = queue.Queue()
        
        # Запускаем рабочий поток для этого типа
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(task_type, processor_func),
            daemon=True
        )
        worker_thread.start()
        self.worker_threads.append(worker_thread)
        
        logger.info(f"Зарегистрирован тип задач: {task_type}")
    
    def add_task(self, task_id: str, data: Any, task_type: str, 
                 priority: int = 0) -> bool:
        """
        Добавляет задачу в очередь
        
        Args:
            task_id: ID задачи
            data: Данные задачи
            task_type: Тип задачи
            priority: Приоритет (больше = выше приоритет)
            
        Returns:
            True если задача добавлена успешно
        """
        if task_type not in self.task_queues:
            logger.error(f"Неизвестный тип задачи: {task_type}")
            return False
        
        try:
            task = BatchTask(
                task_id=task_id,
                data=data,
                task_type=task_type,
                priority=priority
            )
            
            # Добавляем в очередь с приоритетом
            self.task_queues[task_type].put((-priority, task), block=False)
            
            with self.lock:
                self.stats['total_tasks'] += 1
                self.stats['queue_sizes'][task_type] = self.task_queues[task_type].qsize()
            
            return True
            
        except queue.Full:
            logger.warning(f"Очередь {task_type} переполнена")
            return False
    
    def get_result(self, task_type: str, timeout: float = 5.0) -> Optional[Any]:
        """
        Получает результат обработки
        
        Args:
            task_type: Тип задачи
            timeout: Таймаут ожидания
            
        Returns:
            Результат или None
        """
        try:
            return self.result_queues[task_type].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_batch_results(self, task_type: str, count: int = None) -> List[Any]:
        """
        Получает несколько результатов
        
        Args:
            task_type: Тип задачи
            count: Количество результатов (None = все доступные)
            
        Returns:
            Список результатов
        """
        results = []
        max_count = count or float('inf')
        
        while len(results) < max_count:
            try:
                result = self.result_queues[task_type].get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def _worker_loop(self, task_type: str, processor_func: Callable):
        """Основной цикл рабочего потока"""
        logger.info(f"Запущен рабочий поток для {task_type}")
        
        batch = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Собираем батч
                current_time = time.time()
                time_elapsed = current_time - last_batch_time
                
                # Проверяем условия для обработки батча
                should_process = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and time_elapsed >= self.max_wait_time)
                )
                
                if should_process and batch:
                    # Обрабатываем батч
                    self._process_batch(task_type, batch, processor_func)
                    batch = []
                    last_batch_time = current_time
                
                # Добавляем новую задачу в батч
                try:
                    priority, task = self.task_queues[task_type].get(timeout=0.1)
                    batch.append(task)
                except queue.Empty:
                    continue
                
            except Exception as e:
                logger.error(f"Ошибка в рабочем потоке {task_type}: {e}")
                time.sleep(1)
    
    def _process_batch(self, task_type: str, batch: List[BatchTask], 
                      processor_func: Callable):
        """Обрабатывает батч задач"""
        start_time = time.time()
        
        try:
            # Извлекаем данные из задач
            task_data = [(task.task_id, task.data) for task in batch]
            
            # Обрабатываем батч
            results = processor_func(task_data)
            
            # Сохраняем результаты
            if isinstance(results, list):
                for result in results:
                    self.result_queues[task_type].put(result)
            else:
                self.result_queues[task_type].put(results)
            
            # Обновляем статистику
            processing_time = time.time() - start_time
            with self.lock:
                self.stats['processed_tasks'] += len(batch)
                self.stats['avg_batch_time'] = (
                    self.stats['avg_batch_time'] * 0.9 + 
                    processing_time * 0.1
                )
            
            logger.debug(f"Обработан батч {task_type}: {len(batch)} задач за "
                        f"{processing_time:.2f}с")
            
        except Exception as e:
            logger.error(f"Ошибка обработки батча {task_type}: {e}")
            with self.lock:
                self.stats['failed_tasks'] += len(batch)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику процессора"""
        with self.lock:
            return {
                'total_tasks': self.stats['total_tasks'],
                'processed_tasks': self.stats['processed_tasks'],
                'failed_tasks': self.stats['failed_tasks'],
                'avg_batch_time': self.stats['avg_batch_time'],
                'queue_sizes': dict(self.stats['queue_sizes']),
                'active_workers': len(self.worker_threads)
            }
    
    def clear_queue(self, task_type: str):
        """Очищает очередь задач"""
        if task_type in self.task_queues:
            while not self.task_queues[task_type].empty():
                try:
                    self.task_queues[task_type].get_nowait()
                except queue.Empty:
                    break
            
            logger.info(f"Очередь {task_type} очищена")
    
    def shutdown(self):
        """Останавливает процессор"""
        self.running = False
        
        # Ждем завершения всех потоков
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        # Закрываем пул потоков
        self.executor.shutdown(wait=True)
        
        logger.info("BatchProcessor остановлен")


class EmbeddingBatchProcessor:
    """Специализированный процессор для эмбеддингов"""
    
    def __init__(self, embedding_engine, batch_size: int = 32):
        self.embedding_engine = embedding_engine
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            max_wait_time=0.5,
            num_workers=2
        )
        
        # Регистрируем обработчик эмбеддингов
        self.batch_processor.register_task_type(
            'embedding', 
            self._process_embedding_batch
        )
    
    def add_embedding_task(self, task_id: str, text: str) -> bool:
        """Добавляет задачу создания эмбеддинга"""
        return self.batch_processor.add_task(
            task_id=task_id,
            data=text,
            task_type='embedding'
        )
    
    def get_embedding_result(self, task_id: str) -> Optional[Any]:
        """Получает результат создания эмбеддинга"""
        results = self.batch_processor.get_batch_results('embedding')
        
        for result in results:
            if isinstance(result, dict) and result.get('task_id') == task_id:
                return result.get('embedding')
        
        return None
    
    def _process_embedding_batch(self, batch_data: List[Tuple[str, str]]) -> List[Dict]:
        """Обрабатывает батч эмбеддингов"""
        if not batch_data:
            return []
        
        # Извлекаем тексты
        task_ids, texts = zip(*batch_data)
        
        # Создаем эмбеддинги
        embeddings = self.embedding_engine.encode(list(texts))
        
        # Формируем результаты
        results = []
        for task_id, embedding in zip(task_ids, embeddings):
            results.append({
                'task_id': task_id,
                'embedding': embedding,
                'status': 'success'
            })
        
        return results


class FactExtractionBatchProcessor:
    """Специализированный процессор для извлечения фактов"""
    
    def __init__(self, fact_extractor, batch_size: int = 5):
        self.fact_extractor = fact_extractor
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            max_wait_time=2.0,
            num_workers=1  # Факты лучше обрабатывать последовательно
        )
        
        # Регистрируем обработчик фактов
        self.batch_processor.register_task_type(
            'fact_extraction',
            self._process_fact_batch
        )
    
    def add_fact_task(self, task_id: str, text: str, session_id: str, 
                     dialogue_id: str) -> bool:
        """Добавляет задачу извлечения фактов"""
        data = {
            'text': text,
            'session_id': session_id,
            'dialogue_id': dialogue_id
        }
        
        return self.batch_processor.add_task(
            task_id=task_id,
            data=data,
            task_type='fact_extraction'
        )
    
    def get_fact_results(self) -> List[Dict]:
        """Получает результаты извлечения фактов"""
        return self.batch_processor.get_batch_results('fact_extraction')
    
    def _process_fact_batch(self, batch_data: List[Tuple[str, Dict]]) -> List[Dict]:
        """Обрабатывает батч извлечения фактов"""
        results = []
        
        for task_id, data in batch_data:
            try:
                facts = self.fact_extractor.extract_facts_from_text(
                    data['text'],
                    data['session_id'],
                    data['dialogue_id']
                )
                
                results.append({
                    'task_id': task_id,
                    'facts': facts,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Ошибка извлечения фактов для {task_id}: {e}")
                results.append({
                    'task_id': task_id,
                    'facts': [],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
