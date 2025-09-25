#!/usr/bin/env python3
"""
Тест автономности ExtractionModule
Проверяет что модуль работает БЕЗ зависимостей от других модулей
"""

import sys
import logging
from typing import Dict, Any, List

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_module_autonomy():
    """Тестирует что модуль работает полностью автономно"""
    
    print("🔬 ТЕСТ АВТОНОМНОСТИ EXTRACTION MODULE")
    print("=" * 60)
    
    # Импортируем ТОЛЬКО модуль извлечения
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    from src.submit.modules.extraction.module import ExtractionModule
    
    print("\n✅ Шаг 1: Создание модуля БЕЗ зависимостей")
    
    config = {
        'min_confidence': 0.5,
        'use_rules': True,
        'use_llm': False,
        'conflict_strategy': 'latest',
        'filter_copypaste': True
    }
    
    # Создаем модуль
    extractor = ExtractionModule(config)
    print("  ✓ Модуль создан без зависимостей")
    
    # НЕ вызываем set_dependencies - проверяем автономность!
    print("\n✅ Шаг 2: Работа БЕЗ optimizer")
    
    # Тестовые данные
    test_texts = [
        "Привет! Меня зовут Иван Петров, мне 28 лет.",
        "Я работаю программистом в Яндексе.",
        "У меня есть кот Барсик и собака Рекс.",
        "Вчера женился на Марии!",
        "Раньше жил в Москве, теперь переехал в Питер."
    ]
    
    dialogue_id = "test_autonomy"
    facts_total = 0
    
    print("\n📝 Извлечение фактов БЕЗ кэширования:")
    
    for i, text in enumerate(test_texts):
        result = extractor.extract_facts(
            text,
            {'dialogue_id': dialogue_id, 'session_id': f'session_{i}'}
        )
        
        if result.success:
            facts_count = len(result.data)
            facts_total += facts_count
            print(f"  Сессия {i}: извлечено {facts_count} фактов")
            
            # Показываем извлеченные факты
            for fact in result.data[:2]:  # Первые 2
                print(f"    - {fact.type.value}: {fact.object}")
        else:
            print(f"  ❌ Ошибка: {result.error}")
    
    print(f"\n  ✓ Всего извлечено: {facts_total} фактов")
    
    # Проверяем что модуль НЕ падает без optimizer
    assert facts_total > 0, "Модуль должен извлекать факты без optimizer"
    
    print("\n✅ Шаг 3: Поиск фактов БЕЗ кэша")
    
    queries = [
        "Как меня зовут?",
        "Где я работаю?",
        "Какие у меня питомцы?",
        "Женат ли я?"
    ]
    
    for query in queries:
        result = extractor.query_facts(dialogue_id, query)
        
        if result.success and result.data:
            print(f"  ✓ '{query}' -> найдено {len(result.data)} фактов")
            if result.data:
                fact = result.data[0]
                print(f"    Ответ: {fact.object}")
        else:
            print(f"  - '{query}' -> не найдено")
    
    print("\n✅ Шаг 4: Построение профиля БЕЗ внешних модулей")
    
    profile_result = extractor.get_user_profile(dialogue_id)
    
    if profile_result.success:
        profile = profile_result.data
        
        # Проверяем критические факты
        critical = profile.get('critical_facts', {})
        
        print(f"\n  📋 Профиль пользователя:")
        print(f"    Имя: {critical.get('name', 'не найдено')}")
        print(f"    Возраст: {critical.get('age', 'не найден')}")
        print(f"    Работа: {critical.get('occupation', 'не найдена')}")
        print(f"    Статус: {critical.get('marital_status', 'не найден')}")
        
        stats = profile.get('stats', {})
        print(f"\n  📊 Статистика:")
        print(f"    Всего фактов: {stats.get('total_facts', 0)}")
        print(f"    Конфликтов разрешено: {stats.get('conflicts_resolved', 0)}")
    
    print("\n✅ Шаг 5: Timeline изменений БЕЗ зависимостей")
    
    timeline = extractor.get_fact_timeline(dialogue_id, 'personal_location')
    
    if timeline:
        print(f"\n  📅 История местоположения:")
        for entry in timeline:
            status = "текущее" if entry.get('is_current') else "прошлое"
            print(f"    - {entry['value']} ({status})")
    
    print("\n✅ Шаг 6: Проверка изоляции модуля")
    
    # Проверяем что модуль НЕ имеет ссылок на другие модули
    assert not hasattr(extractor, 'storage') or extractor.storage is None
    assert not hasattr(extractor, 'embeddings') or extractor.embeddings is None
    assert not hasattr(extractor, 'model_inference') or extractor.model_inference is None
    
    # Единственная допустимая зависимость - optimizer (и то опциональная)
    assert extractor.optimizer is None, "Optimizer должен быть None без set_dependencies"
    
    print("  ✓ Модуль не имеет зависимостей от других модулей")
    print("  ✓ Модуль полностью автономен")
    
    # Получаем финальную статистику
    stats = extractor.get_stats()
    
    print("\n📈 ФИНАЛЬНАЯ СТАТИСТИКА:")
    print(f"  Извлечено фактов: {stats['module_stats']['total_extracted']}")
    print(f"  Попаданий в кэш: {stats['module_stats']['cache_hits']} (должно быть 0)")
    print(f"  Отфильтровано копипаста: {stats['module_stats']['copypaste_filtered']}")
    
    # Проверяем что кэш не использовался
    assert stats['module_stats']['cache_hits'] == 0, "Кэш не должен работать без optimizer"
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ АВТОНОМНОСТИ ПРОЙДЕНЫ!")
    print("🎉 Модуль работает ПОЛНОСТЬЮ НЕЗАВИСИМО!")


def test_with_optimizer_only():
    """Тестирует работу ТОЛЬКО с optimizer"""
    
    print("\n\n🔧 ТЕСТ С OPTIMIZER (единственная зависимость)")
    print("=" * 60)
    
    from modules.extraction.module import ExtractionModule
    
    # Простой мок Optimizer
    class MockOptimizer:
        def __init__(self):
            self.cache = {}
            self.hits = 0
            self.puts = 0
        
        def cache_get(self, key: str):
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            return None
        
        def cache_put(self, key: str, value: Any, ttl: int = None):
            self.cache[key] = value
            self.puts += 1
    
    # Создаем модуль
    config = {'min_confidence': 0.5, 'use_rules': True}
    extractor = ExtractionModule(config)
    
    # Создаем ТОЛЬКО optimizer
    optimizer = MockOptimizer()
    
    # Устанавливаем ТОЛЬКО optimizer
    print("\n✅ Устанавливаем ТОЛЬКО optimizer")
    extractor.set_dependencies(optimizer=optimizer)
    
    dialogue_id = "test_with_cache"
    
    # Первый запрос - без кэша
    print("\n📝 Первое извлечение (без кэша):")
    result1 = extractor.extract_facts(
        "Меня зовут Петр, мне 35 лет",
        {'dialogue_id': dialogue_id, 'session_id': '1'}
    )
    print(f"  Извлечено: {len(result1.data)} фактов")
    print(f"  Кэш: hits={optimizer.hits}, puts={optimizer.puts}")
    
    # Второй запрос с тем же текстом - из кэша
    print("\n📝 Повторное извлечение (из кэша):")
    result2 = extractor.extract_facts(
        "Меня зовут Петр, мне 35 лет",
        {'dialogue_id': dialogue_id, 'session_id': '1'}
    )
    print(f"  Извлечено: {len(result2.data)} фактов")
    print(f"  Из кэша: {result2.metadata.get('from_cache', False)}")
    print(f"  Кэш: hits={optimizer.hits}, puts={optimizer.puts}")
    
    assert optimizer.hits > 0, "Должны быть попадания в кэш"
    assert result2.metadata.get('from_cache'), "Второй запрос должен быть из кэша"
    
    print("\n✅ Модуль правильно использует ТОЛЬКО optimizer для кэширования")


if __name__ == "__main__":
    try:
        # Тест полной автономности
        test_module_autonomy()
        
        # Тест с optimizer
        test_with_optimizer_only()
        
        print("\n" + "🎉" * 20)
        print("МОДУЛЬ EXTRACTION ПОЛНОСТЬЮ АВТОНОМЕН!")
        print("ГОТОВ К ИНТЕГРАЦИИ В СИСТЕМУ!")
        print("🎉" * 20)
        
    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)