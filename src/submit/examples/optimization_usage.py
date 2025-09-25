# examples/optimization_usage.py
"""
Пример использования расширенного модуля оптимизации
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Импортируем модуль оптимизации
from src.submit.modules.optimization.module import OptimizationModule


def main():
    """Демонстрация возможностей модуля оптимизации"""
    
    # ==========================================
    # 1. ИНИЦИАЛИЗАЦИЯ
    # ==========================================
    print("=" * 50)
    print("1. Инициализация модуля оптимизации")
    print("=" * 50)
    
    config = {
        # Конфигурация L1 кэша (самый быстрый, в памяти)
        'l1_cache_size': 50,
        
        # Конфигурация L2 кэша (основной кэш)
        'l2_cache_size': 1000,
        'l2_cache_memory': 256,  # МБ
        'eviction_strategy': 'lru',
        'default_ttl': 3600,  # 1 час
        
        # Конфигурация L3 кэша (дисковый)
        'disk_cache_path': './cache_storage',
        'l3_cache_max_age_days': 7,
        
        # Конфигурация батчевой обработки
        'batch_size': 32,
        'max_wait': 1.0,
        'num_workers': 4
    }
    
    optimizer = OptimizationModule(config)
    print(f"✅ Модуль инициализирован с 3-уровневым кэшем")
    print(f"   L1: {config['l1_cache_size']} записей")
    print(f"   L2: {config['l2_cache_size']} записей, {config['l2_cache_memory']}MB")
    print(f"   L3: {config['disk_cache_path']}")
    
    # ==========================================
    # 2. МНОГОУРОВНЕВОЕ КЭШИРОВАНИЕ
    # ==========================================
    print("\n" + "=" * 50)
    print("2. Демонстрация многоуровневого кэширования")
    print("=" * 50)
    
    # Добавляем данные в кэш
    test_data = {
        'user_profile': {'id': 123, 'name': 'John', 'preferences': ['tech', 'science']},
        'session_data': {'session_id': 'abc123', 'start_time': time.time()},
        'computed_result': np.random.rand(100, 50).mean()
    }
    
    for key, value in test_data.items():
        optimizer.cache_put(key, value)
        print(f"✅ Добавлено в кэш: {key}")
    
    # Получаем данные (будет из L1)
    print("\n📊 Получение данных из кэша:")
    for key in test_data.keys():
        value = optimizer.cache_get(key)
        if value is not None:
            print(f"   ✓ {key}: получено из кэша")
    
    # Проверяем размеры кэшей
    cache_sizes = optimizer.get_cache_sizes()
    print(f"\n📈 Размеры кэшей:")
    print(f"   L1: {cache_sizes['l1_entries']} записей")
    print(f"   L2: {cache_sizes['l2_entries']} записей")
    print(f"   L3: {cache_sizes['l3_entries']} файлов")
    
    # ==========================================
    # 3. ОПТИМИЗАЦИЯ ДЛЯ РАЗНЫХ ТИПОВ ДАННЫХ
    # ==========================================
    print("\n" + "=" * 50)
    print("3. Оптимизация для специфичных типов данных")
    print("=" * 50)
    
    # Включаем оптимизацию для эмбеддингов
    optimizer.optimize_for_embeddings()
    
    # Создаем и кэшируем эмбеддинги
    embeddings = np.random.rand(1000, 768).astype(np.float32)  # Типичный размер BERT
    print(f"📊 Размер эмбеддингов до оптимизации: {embeddings.nbytes / 1024:.2f} KB")
    
    optimizer.cache_put('embeddings_batch', embeddings)
    print("✅ Эмбеддинги сохранены с оптимизацией (float16)")
    
    # Включаем оптимизацию для текстов
    optimizer.optimize_for_text()
    
    # Создаем и кэшируем большой текст
    large_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 1000
    print(f"📊 Размер текста до сжатия: {len(large_text)} символов")
    
    optimizer.cache_put('large_document', large_text)
    print("✅ Текст сохранен со сжатием")
    
    # ==========================================
    # 4. БАТЧЕВАЯ ОБРАБОТКА С ПРИОРИТЕТАМИ
    # ==========================================
    print("\n" + "=" * 50)
    print("4. Батчевая обработка с приоритетами")
    print("=" * 50)
    
    # Создаем задачи с разными приоритетами
    tasks = []
    
    # Критичные задачи (высокий приоритет)
    for i in range(3):
        tasks.append({
            'id': f'critical_{i}',
            'data': f'Критичная задача {i}',
            'priority': 9,
            'type': 'critical'
        })
    
    # Обычные задачи
    for i in range(10):
        tasks.append({
            'id': f'normal_{i}',
            'data': f'Обычная задача {i}',
            'priority': 5,
            'type': 'normal'
        })
    
    # Низкоприоритетные задачи
    for i in range(5):
        tasks.append({
            'id': f'low_{i}',
            'data': f'Фоновая задача {i}',
            'priority': 1,
            'type': 'background'
        })
    
    print(f"📋 Создано задач:")
    print(f"   Критичных: 3 (приоритет 9)")
    print(f"   Обычных: 10 (приоритет 5)")
    print(f"   Фоновых: 5 (приоритет 1)")
    
    # Функция обработки
    processed_order = []
    
    def process_task(task: Dict) -> Dict:
        """Обрабатывает задачу"""
        processed_order.append(task['id'])
        
        # Симулируем обработку
        if task['type'] == 'critical':
            time.sleep(0.05)  # Быстрая обработка критичных
        else:
            time.sleep(0.01)
        
        return {
            'task_id': task['id'],
            'result': f"Обработано: {task['data']}",
            'processed_at': time.time()
        }
    
    # Запускаем батчевую обработку с приоритетами
    print("\n⚙️ Запуск батчевой обработки...")
    start_time = time.time()
    
    result = optimizer.batch_process_priority(tasks, process_task)
    
    processing_time = time.time() - start_time
    
    if result.success:
        print(f"✅ Обработано {len(result.data)} задач за {processing_time:.2f} сек")
        print(f"   Высокоприоритетных: {result.metadata['high_priority_count']}")
        print(f"   Обычных: {result.metadata['normal_priority_count']}")
        
        # Проверяем порядок обработки
        print("\n📊 Порядок обработки (первые 5):")
        for i, task_id in enumerate(processed_order[:5], 1):
            print(f"   {i}. {task_id}")
    
    # ==========================================
    # 5. МОНИТОРИНГ ПРОИЗВОДИТЕЛЬНОСТИ
    # ==========================================
    print("\n" + "=" * 50)
    print("5. Мониторинг производительности")
    print("=" * 50)
    
    # Симулируем нагрузку
    print("⚙️ Симуляция нагрузки...")
    
    for i in range(100):
        # Кэширование
        key = f"test_key_{i}"
        value = f"test_value_{i}"
        optimizer.cache_put(key, value)
        
        # Чтение (90% попаданий)
        if i % 10 < 9:
            optimizer.cache_get(f"test_key_{i}")
        else:
            optimizer.cache_get(f"missing_key_{i}")
    
    # Получаем отчет о производительности
    perf_report = optimizer.get_performance_report()
    
    if perf_report.success:
        report_data = perf_report.data
        
        print("\n📊 Отчет о производительности:")
        print(f"\n🎯 Кэш:")
        print(f"   Hit Rate: {report_data['cache_hit_rate']:.2%}")
        print(f"   L1 попаданий: {report_data['cache_stats']['l1_hits']}")
        print(f"   L2 попаданий: {report_data['cache_stats']['l2_hits']}")
        print(f"   L3 попаданий: {report_data['cache_stats']['l3_hits']}")
        print(f"   Промахов: {report_data['cache_stats']['misses']}")
        
        print(f"\n📦 Батчевая обработка:")
        batch_stats = report_data['batch_stats']
        print(f"   Обработано элементов: {batch_stats['total_processed']}")
        print(f"   Количество батчей: {batch_stats['batch_count']}")
        print(f"   Среднее время батча: {batch_stats['avg_batch_time_ms']:.2f} мс")
        
        print(f"\n💾 Использование ресурсов:")
        resources = report_data['resource_usage']
        print(f"   Память: {resources['memory_usage_mb']:.2f} MB")
        print(f"   CPU: {resources['cpu_usage_percent']:.1f}%")
        
        # Анализ трендов
        if 'trends' in report_data and report_data['trends']:
            trends = report_data['trends']
            print(f"\n📈 Тренды производительности:")
            print(f"   Тренд: {trends.get('trend', 'N/A')}")
            if 'avg_processing_time' in trends:
                print(f"   Среднее время обработки: {trends['avg_processing_time']:.3f} сек")
    
    # ==========================================
    # 6. ПРОГРЕВ КЭША
    # ==========================================
    print("\n" + "=" * 50)
    print("6. Прогрев кэша начальными данными")
    print("=" * 50)
    
    # Подготавливаем данные для прогрева
    warmup_data = [
        ('config:app', {'version': '1.0', 'features': ['cache', 'batch']}),
        ('user:preferences', {'theme': 'dark', 'language': 'ru'}),
        ('model:weights', np.random.rand(100, 100)),
        ('vocab:tokens', ['token1', 'token2', 'token3']),
        ('stats:global', {'requests': 0, 'errors': 0})
    ]
    
    print(f"📦 Прогрев кэша {len(warmup_data)} записями...")
    optimizer.warmup_cache(warmup_data)
    
    # Проверяем, что данные в кэше
    for key, _ in warmup_data:
        if optimizer.cache_get(key) is not None:
            print(f"   ✓ {key}: успешно загружено")
    
    # ==========================================
    # 7. АВТОМАТИЧЕСКАЯ ОЧИСТКА
    # ==========================================
    print("\n" + "=" * 50)
    print("7. Демонстрация автоматической очистки")
    print("=" * 50)
    
    # Переполняем L1 кэш
    print("⚠️ Переполнение L1 кэша...")
    for i in range(100):  # Намного больше чем L1 размер
        optimizer.cache_put(f'overflow_{i}', f'value_{i}')
    
    # Проверяем, что автоочистка работает
    cache_sizes = optimizer.get_cache_sizes()
    print(f"✅ После переполнения:")
    print(f"   L1 размер: {cache_sizes['l1_entries']} записей")
    print(f"   (автоматически ограничен до {config['l1_cache_size'] * 2})")
    
    # ==========================================
    # 8. РАБОТА С РАЗНЫМИ ТИПАМИ КЭША
    # ==========================================
    print("\n" + "=" * 50)
    print("8. Работа с специализированными типами кэша")
    print("=" * 50)
    
    # Эмбеддинги
    embedding_cache_key = 'user_embedding_123'
    user_embedding = np.random.rand(768).astype(np.float32)
    optimizer.l2_cache.put(embedding_cache_key, user_embedding, cache_type='embedding')
    print(f"✅ Сохранен эмбеддинг пользователя")
    
    # Факты
    facts_cache_key = 'session_facts_abc'
    extracted_facts = [
        {'fact': 'User prefers Python', 'confidence': 0.95},
        {'fact': 'User works with ML', 'confidence': 0.88},
        {'fact': 'User likes optimization', 'confidence': 0.92}
    ]
    optimizer.l2_cache.put(facts_cache_key, extracted_facts, cache_type='fact')
    print(f"✅ Сохранены {len(extracted_facts)} извлеченных фактов")
    
    # Сжатые данные
    compression_cache_key = 'compressed_log'
    log_data = "ERROR " * 1000 + "WARNING " * 500 + "INFO " * 2000
    optimizer.l2_cache.put(compression_cache_key, log_data, cache_type='compression')
    print(f"✅ Сохранены сжатые логи ({len(log_data)} символов)")
    
    # Результаты запросов
    query_cache_key = 'search_results_ml'
    search_results = {
        'query': 'machine learning optimization',
        'results': ['result1', 'result2', 'result3'],
        'timestamp': time.time()
    }
    optimizer.l2_cache.put(query_cache_key, search_results, cache_type='query')
    print(f"✅ Сохранены результаты поиска")
    
    # ==========================================
    # 9. ПАРАЛЛЕЛЬНАЯ БАТЧЕВАЯ ОБРАБОТКА
    # ==========================================
    print("\n" + "=" * 50)
    print("9. Параллельная батчевая обработка")
    print("=" * 50)
    
    # Регистрируем несколько типов обработчиков
    def text_processor(batch_data):
        """Обработчик текстов"""
        results = []
        for task_id, text in batch_data:
            # Симулируем обработку текста
            result = {
                'task_id': task_id,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            results.append(result)
        return results
    
    def number_processor(batch_data):
        """Обработчик чисел"""
        results = []
        for task_id, numbers in batch_data:
            # Симулируем математические вычисления
            result = {
                'task_id': task_id,
                'sum': sum(numbers),
                'mean': sum(numbers) / len(numbers) if numbers else 0
            }
            results.append(result)
        return results
    
    # Регистрируем обработчики
    optimizer.batch_processor.register_task_type('text_processing', text_processor)
    optimizer.batch_processor.register_task_type('number_processing', number_processor)
    
    print("📝 Добавляем задачи разных типов...")
    
    # Добавляем текстовые задачи
    for i in range(10):
        text = f"This is text number {i} with some words to process"
        optimizer.batch_processor.add_task(f'text_{i}', text, 'text_processing')
    
    # Добавляем числовые задачи
    for i in range(10):
        numbers = list(range(i, i + 10))
        optimizer.batch_processor.add_task(f'numbers_{i}', numbers, 'number_processing')
    
    # Ждем обработки
    time.sleep(2)
    
    # Получаем результаты
    text_results = optimizer.batch_processor.get_batch_results('text_processing')
    number_results = optimizer.batch_processor.get_batch_results('number_processing')
    
    print(f"✅ Обработано текстовых задач: {len(text_results)}")
    print(f"✅ Обработано числовых задач: {len(number_results)}")
    
    # Показываем статистику батч-процессора
    batch_stats = optimizer.batch_processor.get_stats()
    print(f"\n📊 Статистика батч-процессора:")
    print(f"   Всего задач: {batch_stats['total_tasks']}")
    print(f"   Обработано: {batch_stats['processed_tasks']}")
    print(f"   Ошибок: {batch_stats['failed_tasks']}")
    print(f"   Среднее время батча: {batch_stats['avg_batch_time']:.3f} сек")
    
    # ==========================================
    # 10. СОХРАНЕНИЕ И ВОССТАНОВЛЕНИЕ
    # ==========================================
    print("\n" + "=" * 50)
    print("10. Сохранение и восстановление состояния")
    print("=" * 50)
    
    # Сохраняем состояние L2 кэша
    cache_backup_path = './cache_backup.pkl'
    optimizer.l2_cache.save(cache_backup_path)
    print(f"💾 L2 кэш сохранен в {cache_backup_path}")
    
    # Получаем финальную статистику
    final_sizes = optimizer.get_cache_sizes()
    final_report = optimizer.get_performance_report()
    
    print(f"\n📊 Финальная статистика:")
    print(f"   L1: {final_sizes['l1_entries']} записей")
    print(f"   L2: {final_sizes['l2_entries']} записей")
    print(f"   L3: {final_sizes['l3_entries']} файлов")
    print(f"   Hit Rate: {final_report.data['cache_hit_rate']:.2%}")
    
    # ==========================================
    # 11. ОЧИСТКА
    # ==========================================
    print("\n" + "=" * 50)
    print("11. Очистка ресурсов")
    print("=" * 50)
    
    # Останавливаем батч-процессор
    optimizer.batch_processor.shutdown()
    print("✅ Батч-процессор остановлен")
    
    # Очищаем все кэши
    optimizer.clear_all_caches()
    print("✅ Все кэши очищены")
    
    # Удаляем временные файлы
    if Path(cache_backup_path).exists():
        Path(cache_backup_path).unlink()
    print("✅ Временные файлы удалены")
    
    print("\n" + "=" * 50)
    print("✅ Демонстрация завершена успешно!")
    print("=" * 50)


def demonstrate_advanced_features():
    """Демонстрация продвинутых возможностей"""
    
    print("\n" + "=" * 50)
    print("ПРОДВИНУТЫЕ ВОЗМОЖНОСТИ")
    print("=" * 50)
    
    config = {
        'l1_cache_size': 20,
        'l2_cache_size': 200,
        'l2_cache_memory': 128,
        'disk_cache_path': './advanced_cache',
        'batch_size': 16,
        'num_workers': 2
    }
    
    optimizer = OptimizationModule(config)
    
    # ==========================================
    # A. АДАПТИВНАЯ ОПТИМИЗАЦИЯ
    # ==========================================
    print("\n📊 A. Адаптивная оптимизация на основе метрик")
    
    def adaptive_optimization():
        """Адаптирует параметры на основе производительности"""
        
        # Получаем текущие метрики
        report = optimizer.get_performance_report()
        if not report.success:
            return
        
        data = report.data
        hit_rate = data['cache_hit_rate']
        
        # Адаптируем размер L1 кэша
        if hit_rate < 0.7:
            # Увеличиваем L1 если низкий hit rate
            optimizer.l1_max_size = min(optimizer.l1_max_size * 1.5, 1000)
            print(f"   📈 L1 размер увеличен до {optimizer.l1_max_size}")
        elif hit_rate > 0.95:
            # Можем уменьшить если очень высокий hit rate
            optimizer.l1_max_size = max(optimizer.l1_max_size * 0.8, 10)
            print(f"   📉 L1 размер уменьшен до {optimizer.l1_max_size}")
        
        # Адаптируем размер батча
        avg_batch_time = data['batch_stats']['avg_batch_time_ms']
        if avg_batch_time > 100:  # Если батчи обрабатываются медленно
            new_batch_size = max(optimizer.batch_processor.batch_size // 2, 4)
            optimizer.batch_processor.batch_size = new_batch_size
            print(f"   📉 Размер батча уменьшен до {new_batch_size}")
        elif avg_batch_time < 20:  # Если очень быстро
            new_batch_size = min(optimizer.batch_processor.batch_size * 2, 128)
            optimizer.batch_processor.batch_size = new_batch_size
            print(f"   📈 Размер батча увеличен до {new_batch_size}")
    
    # Симулируем нагрузку и адаптацию
    for iteration in range(3):
        print(f"\n   Итерация {iteration + 1}:")
        
        # Генерируем нагрузку
        for i in range(50):
            optimizer.cache_put(f'adaptive_{iteration}_{i}', f'value_{i}')
            if i % 3 == 0:
                optimizer.cache_get(f'adaptive_{iteration}_{i}')
        
        # Адаптируем
        adaptive_optimization()
    
    # ==========================================
    # B. КЭШИРОВАНИЕ С ПРЕДСКАЗАНИЕМ
    # ==========================================
    print("\n🔮 B. Предиктивное кэширование")
    
    class PredictiveCache:
        """Кэш с предсказанием следующих запросов"""
        
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.access_patterns = {}
            
        def access(self, key: str):
            """Доступ с предсказанием"""
            # Записываем паттерн
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            self.access_patterns[key].append(time.time())
            
            # Получаем значение
            value = self.optimizer.cache_get(key)
            
            # Предзагружаем связанные данные
            self._prefetch_related(key)
            
            return value
        
        def _prefetch_related(self, key: str):
            """Предзагружает связанные данные"""
            # Простая эвристика: если обращаются к key_N, 
            # вероятно скоро понадобится key_N+1
            if '_' in key:
                prefix, suffix = key.rsplit('_', 1)
                try:
                    next_idx = int(suffix) + 1
                    next_key = f"{prefix}_{next_idx}"
                    
                    # Проверяем, есть ли в L2/L3, и поднимаем в L1
                    if next_key not in self.optimizer.l1_cache:
                        value = self.optimizer.cache_get(next_key)
                        if value is not None:
                            print(f"      🔮 Предзагружен: {next_key}")
                except ValueError:
                    pass
    
    predictive = PredictiveCache(optimizer)
    
    # Демонстрация
    print("   Последовательный доступ с предсказанием:")
    for i in range(5):
        key = f"sequential_{i}"
        optimizer.cache_put(key, f"data_{i}")
    
    # Обращаемся к первому элементу
    predictive.access("sequential_0")
    # Следующий должен быть предзагружен
    predictive.access("sequential_1")
    
    # ==========================================
    # C. РАСПРЕДЕЛЕННОЕ КЭШИРОВАНИЕ (СИМУЛЯЦИЯ)
    # ==========================================
    print("\n🌐 C. Симуляция распределенного кэширования")
    
    class DistributedCache:
        """Симуляция распределенного кэша"""
        
        def __init__(self, nodes: int = 3):
            self.nodes = []
            for i in range(nodes):
                config_node = config.copy()
                config_node['disk_cache_path'] = f'./cache_node_{i}'
                self.nodes.append(OptimizationModule(config_node))
            print(f"   ✅ Создано {nodes} узлов кэша")
        
        def _hash_to_node(self, key: str) -> int:
            """Определяет узел по хешу ключа"""
            return hash(key) % len(self.nodes)
        
        def put(self, key: str, value: Any):
            """Сохраняет в соответствующий узел"""
            node_idx = self._hash_to_node(key)
            self.nodes[node_idx].cache_put(key, value)
            print(f"      → Сохранено на узел {node_idx}: {key}")
        
        def get(self, key: str) -> Any:
            """Получает из соответствующего узла"""
            node_idx = self._hash_to_node(key)
            value = self.nodes[node_idx].cache_get(key)
            if value is not None:
                print(f"      ← Получено с узла {node_idx}: {key}")
            return value
        
        def get_total_stats(self) -> Dict:
            """Собирает статистику со всех узлов"""
            total_entries = 0
            total_hits = 0
            total_misses = 0
            
            for i, node in enumerate(self.nodes):
                report = node.get_performance_report()
                if report.success:
                    stats = report.data['cache_stats']
                    total_hits += stats.get('hits', 0)
                    total_misses += stats.get('misses', 0)
                    
                    sizes = node.get_cache_sizes()
                    total_entries += sizes['l1_entries'] + sizes['l2_entries']
            
            return {
                'total_entries': total_entries,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            }
    
    distributed = DistributedCache(nodes=3)
    
    # Добавляем данные
    print("\n   Распределение данных по узлам:")
    for i in range(10):
        distributed.put(f'distributed_{i}', f'value_{i}')
    
    # Получаем данные
    print("\n   Получение распределенных данных:")
    for i in range(0, 10, 3):
        distributed.get(f'distributed_{i}')
    
    # Статистика
    dist_stats = distributed.get_total_stats()
    print(f"\n   📊 Общая статистика распределенного кэша:")
    print(f"      Всего записей: {dist_stats['total_entries']}")
    print(f"      Hit Rate: {dist_stats['hit_rate']:.2%}")
    
    # Очистка
    for node in distributed.nodes:
        node.clear_all_caches()
    
    print("\n" + "=" * 50)
    print("✅ Продвинутые возможности продемонстрированы!")
    print("=" * 50)


if __name__ == "__main__":
    # Запускаем основную демонстрацию
    main()
    
    # Опционально: демонстрация продвинутых возможностей
    demonstrate_advanced_features()