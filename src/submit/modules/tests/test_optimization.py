# tests/test_optimization.py

import unittest
import tempfile
import time
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Импортируем модули для тестирования
from src.submit.modules.optimization.module import (
    OptimizationModule,
    PerformanceMonitor,
    TypeHandler
)
from src.submit.modules.optimization.cache_manager import CacheManager, CacheEntry
from src.submit.modules.optimization.batch_processor import BatchProcessor, BatchTask


class TestCacheManager(unittest.TestCase):
    """Тесты для менеджера кэширования"""
    
    def setUp(self):
        self.cache = CacheManager(
            max_size=100,
            max_memory_mb=10,
            eviction_strategy='lru',
            ttl_seconds=60
        )
    
    def test_basic_operations(self):
        """Тест базовых операций кэша"""
        # Put и Get
        self.cache.put('key1', 'value1')
        self.assertEqual(self.cache.get('key1'), 'value1')
        
        # Несуществующий ключ
        self.assertIsNone(self.cache.get('nonexistent'))
        
        # Обновление значения
        self.cache.put('key1', 'new_value')
        self.assertEqual(self.cache.get('key1'), 'new_value')
    
    def test_ttl_expiration(self):
        """Тест истечения TTL"""
        # Добавляем с коротким TTL
        self.cache.put('temp_key', 'temp_value', ttl=0)
        time.sleep(0.1)
        
        # Должен быть просрочен
        self.assertIsNone(self.cache.get('temp_key'))
    
    def test_eviction_strategies(self):
        """Тест стратегий вытеснения"""
        # LRU стратегия
        cache_lru = CacheManager(max_size=3, eviction_strategy='lru')
        
        cache_lru.put('a', 1)
        cache_lru.put('b', 2)
        cache_lru.put('c', 3)
        
        # Обращаемся к 'a', теперь 'b' самый старый
        cache_lru.get('a')
        
        # Добавляем новый, 'b' должен быть вытеснен
        cache_lru.put('d', 4)
        
        self.assertIsNone(cache_lru.get('b'))
        self.assertEqual(cache_lru.get('a'), 1)
        self.assertEqual(cache_lru.get('c'), 3)
        self.assertEqual(cache_lru.get('d'), 4)
    
    def test_cache_types(self):
        """Тест различных типов кэша"""
        # Эмбеддинги
        embedding = np.array([1.0, 2.0, 3.0])
        self.cache.put('emb1', embedding, cache_type='embedding')
        np.testing.assert_array_equal(
            self.cache.get('emb1', cache_type='embedding'),
            embedding
        )
        
        # Факты
        facts = [{'fact': 'test', 'confidence': 0.9}]
        self.cache.put('facts1', facts, cache_type='fact')
        self.assertEqual(self.cache.get('facts1', cache_type='fact'), facts)
        
        # Сжатие
        self.cache.put('comp1', 'compressed_data', cache_type='compression')
        self.assertEqual(
            self.cache.get('comp1', cache_type='compression'),
            'compressed_data'
        )
    
    def test_get_or_compute(self):
        """Тест get_or_compute функции"""
        compute_calls = 0
        
        def compute_fn():
            nonlocal compute_calls
            compute_calls += 1
            return 'computed_value'
        
        # Первый вызов должен вычислить
        result = self.cache.get_or_compute('key', compute_fn)
        self.assertEqual(result, 'computed_value')
        self.assertEqual(compute_calls, 1)
        
        # Второй вызов должен взять из кэша
        result = self.cache.get_or_compute('key', compute_fn)
        self.assertEqual(result, 'computed_value')
        self.assertEqual(compute_calls, 1)  # Не увеличилось
    
    def test_cache_stats(self):
        """Тест статистики кэша"""
        self.cache.put('key1', 'value1')
        self.cache.get('key1')  # Hit
        self.cache.get('key2')  # Miss
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 0.5)
    
    def test_save_and_load(self):
        """Тест сохранения и загрузки кэша"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Сохраняем
            self.cache.put('key1', 'value1')
            self.cache.put('key2', 'value2')
            self.cache.save(tmp_path)
            
            # Загружаем в новый кэш
            new_cache = CacheManager()
            new_cache.load(tmp_path)
            
            self.assertEqual(new_cache.get('key1'), 'value1')
            self.assertEqual(new_cache.get('key2'), 'value2')
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestBatchProcessor(unittest.TestCase):
    """Тесты для батч-процессора"""
    
    def setUp(self):
        self.processor = BatchProcessor(
            batch_size=3,
            max_wait_time=0.5,
            num_workers=2
        )
    
    def test_batch_processing(self):
        """Тест батчевой обработки"""
        results = []
        
        def process_batch(batch_data):
            return [f"processed_{task_id}" for task_id, _ in batch_data]
        
        # Регистрируем процессор
        self.processor.register_task_type('test', process_batch)
        
        # Добавляем задачи
        for i in range(5):
            self.processor.add_task(f'task_{i}', f'data_{i}', 'test')
        
        # Ждем обработки
        time.sleep(1)
        
        # Получаем результаты
        batch_results = self.processor.get_batch_results('test', count=5)
        self.assertEqual(len(batch_results), 5)
    
    def test_priority_processing(self):
        """Тест обработки с приоритетами"""
        processed_order = []
        
        def process_batch(batch_data):
            for task_id, data in batch_data:
                processed_order.append(task_id)
            return batch_data
        
        self.processor.register_task_type('priority_test', process_batch)
        
        # Добавляем задачи с разными приоритетами
        self.processor.add_task('low_priority', 'data', 'priority_test', priority=1)
        self.processor.add_task('high_priority', 'data', 'priority_test', priority=10)
        self.processor.add_task('medium_priority', 'data', 'priority_test', priority=5)
        
        # Ждем обработки
        time.sleep(1)
        
        # Высокоприоритетные должны быть обработаны первыми
        self.assertEqual(processed_order[0], 'high_priority')
    
    def test_queue_overflow(self):
        """Тест переполнения очереди"""
        processor = BatchProcessor(max_queue_size=2)
        processor.register_task_type('overflow_test', lambda x: x)
        
        # Заполняем очередь
        self.assertTrue(processor.add_task('task1', 'data', 'overflow_test'))
        self.assertTrue(processor.add_task('task2', 'data', 'overflow_test'))
        
        # Следующая задача не должна быть добавлена
        self.assertFalse(processor.add_task('task3', 'data', 'overflow_test'))
    
    def test_processor_stats(self):
        """Тест статистики процессора"""
        def dummy_processor(batch):
            time.sleep(0.1)
            return batch
        
        self.processor.register_task_type('stats_test', dummy_processor)
        
        # Добавляем задачи
        for i in range(5):
            self.processor.add_task(f'task_{i}', f'data_{i}', 'stats_test')
        
        # Ждем обработки
        time.sleep(1)
        
        stats = self.processor.get_stats()
        self.assertEqual(stats['total_tasks'], 5)
        self.assertGreater(stats['processed_tasks'], 0)
        self.assertGreater(stats['avg_batch_time'], 0)


class TestOptimizationModule(unittest.TestCase):
    """Тесты для модуля оптимизации"""
    
    def setUp(self):
        # Создаем временную директорию для L3 кэша
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = {
            'l1_cache_size': 10,
            'l2_cache_size': 100,
            'l2_cache_memory': 10,
            'disk_cache_path': self.temp_dir,
            'batch_size': 5,
            'max_wait': 0.5
        }
        
        self.module = OptimizationModule(self.config)
    
    def tearDown(self):
        # Очищаем временную директорию
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_multilevel_cache(self):
        """Тест многоуровневого кэширования"""
        # Добавляем в кэш
        self.module.cache_put('test_key', 'test_value')
        
        # Должно быть в L1
        self.assertIn('test_key', self.module.l1_cache)
        
        # Получаем из кэша (должно быть из L1)
        value = self.module.cache_get('test_key')
        self.assertEqual(value, 'test_value')
        
        # Проверяем статистику
        report = self.module.get_performance_report()
        self.assertGreater(report.data['cache_stats']['l1_hits'], 0)
    
    def test_l3_disk_cache(self):
        """Тест дискового кэша (L3)"""
        # Добавляем в кэш
        self.module.cache_put('disk_key', 'disk_value')
        
        # Ждем пока запишется на диск
        time.sleep(0.5)
        
        # Проверяем, что файл создан
        cache_files = list(Path(self.temp_dir).glob("*.pkl"))
        self.assertGreater(len(cache_files), 0)
        
        # Очищаем L1 и L2
        self.module.l1_cache.clear()
        self.module.l2_cache.clear()
        
        # Должно быть получено из L3
        value = self.module.cache_get('disk_key')
        self.assertEqual(value, 'disk_value')
    
    def test_priority_batch_processing(self):
        """Тест батчевой обработки с приоритетами"""
        processed_tasks = []
        
        def processor(task):
            processed_tasks.append(task['id'])
            return task
        
        tasks = [
            {'id': 'low1', 'priority': 1},
            {'id': 'high1', 'priority': 9},
            {'id': 'low2', 'priority': 2},
            {'id': 'high2', 'priority': 8},
            {'id': 'medium', 'priority': 5}
        ]
        
        result = self.module.batch_process_priority(tasks, processor)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 5)
        
        # Проверяем, что высокоприоритетные обработаны первыми
        high_priority_indices = [processed_tasks.index('high1'), 
                                 processed_tasks.index('high2')]
        low_priority_indices = [processed_tasks.index('low1'), 
                               processed_tasks.index('low2')]
        
        self.assertLess(max(high_priority_indices), min(low_priority_indices))
    
    def test_performance_monitoring(self):
        """Тест мониторинга производительности"""
        # Выполняем операции
        for i in range(10):
            self.module.cache_put(f'key_{i}', f'value_{i}')
            self.module.cache_get(f'key_{i}')
        
        # Несуществующие ключи для промахов
        for i in range(5):
            self.module.cache_get(f'missing_{i}')
        
        # Получаем отчет
        report = self.module.get_performance_report()
        
        self.assertTrue(report.success)
        self.assertIn('cache_hit_rate', report.data)
        self.assertIn('cache_stats', report.data)
        self.assertIn('resource_usage', report.data)
        
        # Проверяем корректность hit rate
        cache_stats = report.data['cache_stats']
        hit_rate = report.data['cache_hit_rate']
        self.assertGreater(hit_rate, 0)
        self.assertLess(hit_rate, 1)
    
    def test_type_optimization(self):
        """Тест оптимизации для разных типов данных"""
        # Оптимизация для эмбеддингов
        self.module.optimize_for_embeddings()
        
        embedding = np.random.rand(100).astype(np.float32)
        self.module.cache_put('embedding_key', embedding)
        
        # Проверяем, что эмбеддинг сохранен
        retrieved = self.module.cache_get('embedding_key')
        np.testing.assert_array_almost_equal(retrieved, embedding, decimal=3)
        
        # Оптимизация для текстов
        self.module.optimize_for_text()
        
        long_text = "This is a long text " * 100
        self.module.cache_put('text_key', long_text)
        
        retrieved_text = self.module.cache_get('text_key')
        self.assertEqual(retrieved_text, long_text)
    
    def test_auto_cleanup(self):
        """Тест автоматической очистки"""
        # Переполняем L1 кэш
        for i in range(30):  # Больше чем l1_max_size * 2
            self.module.cache_put(f'overflow_{i}', f'value_{i}')
        
        # L1 не должен превышать определенный размер
        self.assertLessEqual(len(self.module.l1_cache), 
                           self.module.l1_max_size * 2)
    
    def test_cache_warmup(self):
        """Тест прогрева кэша"""
        warmup_data = [
            ('warm1', 'value1'),
            ('warm2', 'value2'),
            ('warm3', 'value3')
        ]
        
        self.module.warmup_cache(warmup_data)
        
        # Все данные должны быть в кэше
        for key, value in warmup_data:
            self.assertEqual(self.module.cache_get(key), value)
    
    def test_concurrent_access(self):
        """Тест конкурентного доступа"""
        import threading
        
        def cache_operations():
            for i in range(100):
                self.module.cache_put(f'concurrent_{i}', f'value_{i}')
                self.module.cache_get(f'concurrent_{i}')
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=cache_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Проверяем, что модуль не сломался
        report = self.module.get_performance_report()
        self.assertTrue(report.success)
    
    def test_clear_all_caches(self):
        """Тест полной очистки кэшей"""
        # Заполняем все уровни
        for i in range(5):
            self.module.cache_put(f'clear_key_{i}', f'value_{i}')
        
        time.sleep(0.5)  # Ждем записи на диск
        
        # Очищаем
        self.module.clear_all_caches()
        
        # Проверяем, что все пусто
        self.assertEqual(len(self.module.l1_cache), 0)
        
        cache_sizes = self.module.get_cache_sizes()
        self.assertEqual(cache_sizes['l1_entries'], 0)
        self.assertEqual(cache_sizes['l2_entries'], 0)


class TestPerformanceMonitor(unittest.TestCase):
    """Тесты для монитора производительности"""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_cache_access_recording(self):
        """Тест записи обращений к кэшу"""
        self.monitor.record_cache_access(True, 'L1')
        self.monitor.record_cache_access(True, 'L2')
        self.monitor.record_cache_access(False)
        
        self.assertEqual(self.monitor.metrics.cache_hits, 2)
        self.assertEqual(self.monitor.metrics.cache_misses, 1)
        self.assertEqual(self.monitor.metrics.l1_hits, 1)
        self.assertEqual(self.monitor.metrics.l2_hits, 1)
    
    def test_batch_recording(self):
        """Тест записи батчевой обработки"""
        self.monitor.record_batch(10, 0.5)
        self.monitor.record_batch(20, 0.7)
        
        self.assertEqual(self.monitor.metrics.batch_processed, 30)
        self.assertEqual(self.monitor.metrics.batch_count, 2)
        self.assertAlmostEqual(self.monitor.metrics.avg_batch_time, 0.6, places=1)
    
    def test_report_generation(self):
        """Тест генерации отчета"""
        # Добавляем данные
        self.monitor.record_cache_access(True, 'L1')
        self.monitor.record_cache_access(False)
        self.monitor.record_batch(15, 0.3)
        
        report = self.monitor.get_report()
        
        self.assertIn('cache_hit_rate', report)
        self.assertIn('cache_stats', report)
        self.assertIn('batch_stats', report)
        self.assertIn('resource_usage', report)
        self.assertEqual(report['cache_hit_rate'], 0.5)
    
    def test_trend_analysis(self):
        """Тест анализа трендов"""
        # Добавляем данные для истории
        for i in range(20):
            processing_time = 0.1 + i * 0.01  # Ухудшающийся тренд
            self.monitor.record_batch(10, processing_time)
        
        trends = self.monitor.get_trends()
        
        self.assertIn('avg_processing_time', trends)
        self.assertIn('trend', trends)
        # Тренд должен показывать ухудшение
        self.assertEqual(trends['trend'], 'degrading')


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'l1_cache_size': 10,
            'l2_cache_size': 100,
            'disk_cache_path': self.temp_dir,
            'batch_size': 5
        }
        self.module = OptimizationModule(self.config)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Тест полного рабочего процесса"""
        # 1. Прогреваем кэш
        warmup_data = [(f'init_{i}', f'value_{i}') for i in range(5)]
        self.module.warmup_cache(warmup_data)
        
        # 2. Выполняем батчевую обработку
        tasks = [
            {'id': f'task_{i}', 'data': f'data_{i}', 'priority': i % 10}
            for i in range(20)
        ]
        
        def process_task(task):
            # Используем кэш во время обработки
            cache_key = f"processed_{task['id']}"
            cached = self.module.cache_get(cache_key)
            if not cached:
                result = f"processed_{task['data']}"
                self.module.cache_put(cache_key, result)
                return result
            return cached
        
        result = self.module.batch_process_priority(tasks, process_task)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 20)
        
        # 3. Проверяем производительность
        report = self.module.get_performance_report()
        
        self.assertGreater(report.data['cache_stats']['hits'], 0)
        self.assertGreater(report.data['batch_stats']['total_processed'], 0)
        
        # 4. Проверяем размеры кэшей
        sizes = self.module.get_cache_sizes()
        self.assertGreater(sizes['l1_entries'], 0)
        
        # 5. Очищаем и проверяем
        self.module.clear_all_caches()
        sizes_after = self.module.get_cache_sizes()
        self.assertEqual(sizes_after['l1_entries'], 0)


if __name__ == '__main__':
    unittest.main()