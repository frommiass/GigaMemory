#!/usr/bin/env python3
"""
Тест для ExtractionModule - проверка всех методов интерфейса IFactExtractor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from submit.modules.extraction.module import ExtractionModule

def test_extraction_module():
    """Тестирует все методы ExtractionModule"""
    
    # Конфигурация
    config = {
        'use_rules': True,
        'conflict_strategy': 'latest',
        'min_confidence': 0.5
    }
    
    # Создаем модуль
    extraction = ExtractionModule(config)
    
    # Тестовые данные
    dialogue_id = "test_dialogue_1"
    text = "Меня зовут Иван, мне 25 лет, я работаю программистом в Москве. Люблю пиццу и кофе."
    context = {
        'dialogue_id': dialogue_id,
        'session_id': 'session_1'
    }
    
    print("🧪 Тестирование ExtractionModule...")
    
    # 1. Тест extract_facts
    print("\n1. Тестируем extract_facts...")
    result = extraction.extract_facts(text, context)
    assert result.success, f"extract_facts failed: {result.error}"
    assert result.data is not None, "extract_facts returned None data"
    print(f"   ✅ extract_facts: извлечено {len(result.data)} фактов")
    
    # 2. Тест get_user_profile
    print("\n2. Тестируем get_user_profile...")
    result = extraction.get_user_profile(dialogue_id)
    assert result.success, f"get_user_profile failed: {result.error}"
    assert result.data is not None, "get_user_profile returned None data"
    print(f"   ✅ get_user_profile: профиль создан")
    
    # 3. Тест query_facts
    print("\n3. Тестируем query_facts...")
    query = "имя"
    result = extraction.query_facts(dialogue_id, query)
    assert result.success, f"query_facts failed: {result.error}"
    assert result.data is not None, "query_facts returned None data"
    print(f"   ✅ query_facts: найдено {len(result.data)} фактов по запросу '{query}'")
    
    # 4. Тест с пустым текстом
    print("\n4. Тестируем extract_facts с пустым текстом...")
    result = extraction.extract_facts("", context)
    assert result.success, f"extract_facts failed with empty text: {result.error}"
    print(f"   ✅ extract_facts: корректно обработал пустой текст")
    
    # 5. Тест с несуществующим диалогом
    print("\n5. Тестируем query_facts для несуществующего диалога...")
    result = extraction.query_facts("nonexistent_dialogue", query)
    assert result.success, f"query_facts failed for nonexistent dialogue: {result.error}"
    assert len(result.data) == 0, "Nonexistent dialogue should return empty facts"
    print(f"   ✅ query_facts: корректно обработал несуществующий диалог")
    
    # 6. Тест с несколькими текстами
    print("\n6. Тестируем extract_facts с несколькими текстами...")
    texts = [
        "Я живу в Санкт-Петербурге",
        "Мне нравится читать книги",
        "Я учусь в университете"
    ]
    
    for i, test_text in enumerate(texts):
        context['session_id'] = f'session_{i+2}'
        result = extraction.extract_facts(test_text, context)
        assert result.success, f"extract_facts failed for text {i+1}: {result.error}"
        print(f"   ✅ extract_facts: текст {i+1} обработан")
    
    # 7. Тест query_facts с разными запросами
    print("\n7. Тестируем query_facts с разными запросами...")
    queries = ["возраст", "работа", "город", "хобби"]
    
    for query in queries:
        result = extraction.query_facts(dialogue_id, query)
        assert result.success, f"query_facts failed for query '{query}': {result.error}"
        print(f"   ✅ query_facts: запрос '{query}' обработан")
    
    print("\n🎉 Все тесты ExtractionModule прошли успешно!")
    print("\n📋 Проверочный чеклист:")
    print("✅ IFactExtractor.extract_facts - работает")
    print("✅ IFactExtractor.get_user_profile - работает") 
    print("✅ IFactExtractor.query_facts - работает")
    print("✅ Обработка ошибок - работает")
    print("✅ Граничные случаи - работают")

if __name__ == "__main__":
    test_extraction_module()
