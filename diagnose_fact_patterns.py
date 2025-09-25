#!/usr/bin/env python3
"""
Диагностика проблемы с FACT_PATTERNS
Находит где именно возникает ошибка PREFERENCE_FOOD
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def diagnose_fact_patterns():
    """Диагностирует структуру FACT_PATTERNS"""
    
    print("🔍 ДИАГНОСТИКА FACT_PATTERNS")
    print("=" * 60)
    
    try:
        # Пробуем импортировать
        from src.submit.modules.extraction.fact_models import FactType
        from src.submit.modules.extraction.fact_patterns import FACT_PATTERNS
        
        print("\n✅ Импорт успешен")
        
        # Проверяем структуру
        print(f"\nВсего паттернов: {len(FACT_PATTERNS)}")
        
        # Проверяем каждый ключ
        print("\n📋 Проверка ключей:")
        
        errors = []
        warnings = []
        
        for i, (key, patterns) in enumerate(FACT_PATTERNS.items()):
            try:
                # Проверяем тип ключа
                if not isinstance(key, FactType):
                    errors.append(f"  ❌ Ключ #{i}: {key} - НЕ является FactType enum!")
                    print(f"  ❌ {key} - не FactType, а {type(key)}")
                    
                    # Пробуем найти соответствующий FactType
                    if isinstance(key, str):
                        # Проверяем есть ли такой тип
                        found = False
                        for ft in FactType:
                            if ft.value == key or ft.name == key:
                                print(f"     → Должен быть: FactType.{ft.name}")
                                found = True
                                break
                        
                        if not found:
                            print(f"     → НЕ НАЙДЕН соответствующий FactType!")
                    continue
                
                # Проверяем паттерны
                if not isinstance(patterns, list):
                    warnings.append(f"  ⚠️  {key.value}: паттерны не список")
                elif len(patterns) == 0:
                    warnings.append(f"  ⚠️  {key.value}: пустой список паттернов")
                else:
                    # Все хорошо с этим типом
                    pass
                    
            except Exception as e:
                errors.append(f"  ❌ Ошибка при проверке ключа #{i}: {e}")
        
        # Выводим результаты
        if errors:
            print("\n❌ НАЙДЕНЫ КРИТИЧЕСКИЕ ОШИБКИ:")
            for error in errors:
                print(error)
        else:
            print("\n✅ Все ключи являются FactType enum")
        
        if warnings:
            print("\n⚠️  Предупреждения:")
            for warning in warnings:
                print(warning)
        
        # Проверяем конкретно PREFERENCE_FOOD
        print("\n🔍 Поиск PREFERENCE_FOOD:")
        
        # Проверяем в FactType
        preference_food_exists = False
        for ft in FactType:
            if ft.name == 'PREFERENCE_FOOD':
                preference_food_exists = True
                print(f"  ✅ FactType.PREFERENCE_FOOD существует: {ft.value}")
                break
        
        if not preference_food_exists:
            print("  ❌ FactType.PREFERENCE_FOOD НЕ НАЙДЕН!")
        
        # Проверяем в FACT_PATTERNS
        preference_food_in_patterns = False
        for key in FACT_PATTERNS:
            if isinstance(key, FactType) and key.name == 'PREFERENCE_FOOD':
                preference_food_in_patterns = True
                print(f"  ✅ PREFERENCE_FOOD есть в FACT_PATTERNS")
                print(f"     Паттернов: {len(FACT_PATTERNS[key])}")
                break
        
        if not preference_food_in_patterns:
            print("  ⚠️  PREFERENCE_FOOD отсутствует в FACT_PATTERNS")
            
            # Проверяем может он там как строка
            for key in FACT_PATTERNS:
                if str(key) == 'PREFERENCE_FOOD' or (hasattr(key, 'value') and key.value == 'preference_food'):
                    print(f"  ❌ Найден как {type(key)}: {key}")
                    break
        
        # Тестовое извлечение
        print("\n🧪 Тест извлечения с PREFERENCE_FOOD:")
        
        test_text = "Я люблю пиццу и суши"
        
        try:
            from src.submit.modules.extraction.fact_extractor import RuleBasedFactExtractor
            
            extractor = RuleBasedFactExtractor()
            facts = extractor.extract_facts_from_text(
                test_text, 
                session_id="test",
                dialogue_id="test"
            )
            
            print(f"  ✅ Извлечено {len(facts)} фактов без ошибок")
            
            for fact in facts:
                if 'food' in fact.type.value.lower() or 'preference' in fact.type.value.lower():
                    print(f"     - {fact.type.value}: {fact.object}")
                    
        except Exception as e:
            print(f"  ❌ Ошибка при извлечении: {e}")
            import traceback
            traceback.print_exc()
        
        return len(errors) == 0
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_fact_patterns():
    """Пытается исправить FACT_PATTERNS"""
    
    print("\n\n🔧 ПОПЫТКА ИСПРАВЛЕНИЯ")
    print("=" * 60)
    
    try:
        from src.submit.modules.extraction.fact_models import FactType
        import src.submit.modules.extraction.fact_patterns as fp_module
        
        # Проверяем и исправляем
        fixed_patterns = {}
        fixes_made = 0
        
        for key, patterns in fp_module.FACT_PATTERNS.items():
            if isinstance(key, FactType):
                # Уже правильный тип
                fixed_patterns[key] = patterns
            elif isinstance(key, str):
                # Пробуем преобразовать строку в FactType
                fixed = False
                
                # Пробуем найти по value
                for ft in FactType:
                    if ft.value == key:
                        fixed_patterns[ft] = patterns
                        fixes_made += 1
                        print(f"  ✅ Исправлено: '{key}' → FactType.{ft.name}")
                        fixed = True
                        break
                
                # Пробуем найти по name
                if not fixed:
                    for ft in FactType:
                        if ft.name == key:
                            fixed_patterns[ft] = patterns
                            fixes_made += 1
                            print(f"  ✅ Исправлено: '{key}' → FactType.{ft.name}")
                            fixed = True
                            break
                
                if not fixed:
                    print(f"  ❌ Не удалось исправить: {key}")
                    # Пропускаем неизвестные ключи
            else:
                print(f"  ⚠️  Неизвестный тип ключа: {type(key)}")
        
        if fixes_made > 0:
            # Заменяем FACT_PATTERNS
            fp_module.FACT_PATTERNS = fixed_patterns
            print(f"\n✅ Исправлено {fixes_made} ключей")
            print("   FACT_PATTERNS обновлен в памяти")
            
            # Проверяем исправление
            print("\n🧪 Проверка после исправления:")
            
            from src.submit.modules.extraction.fact_extractor import RuleBasedFactExtractor
            
            extractor = RuleBasedFactExtractor()
            facts = extractor.extract_facts_from_text(
                "Меня зовут Петр, я люблю пиццу",
                session_id="test",
                dialogue_id="test"  
            )
            
            print(f"  ✅ Извлечено {len(facts)} фактов после исправления")
            
            return True
        else:
            print("\n✅ Исправления не требуются")
            return True
            
    except Exception as e:
        print(f"\n❌ Ошибка при исправлении: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Диагностируем
    success = diagnose_fact_patterns()
    
    if not success:
        # Пробуем исправить
        if fix_fact_patterns():
            print("\n" + "=" * 60)
            print("✅ ПРОБЛЕМА ИСПРАВЛЕНА!")
            print("Теперь можно запускать тесты")
        else:
            print("\n" + "=" * 60)
            print("❌ ТРЕБУЕТСЯ РУЧНОЕ ИСПРАВЛЕНИЕ")
            print("\nВ файле fact_patterns.py замените:")
            print("  'PREFERENCE_FOOD': [...]")
            print("На:")
            print("  FactType.PREFERENCE_FOOD: [...]")
    else:
        print("\n" + "=" * 60)
        print("✅ ПРОБЛЕМ НЕ ОБНАРУЖЕНО!")
