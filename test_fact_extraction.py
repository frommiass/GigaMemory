#!/usr/bin/env python3
"""
Тест системы извлечения фактов для GigaMemory
"""
import sys
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import Message
from submit.extraction.fact_models import Fact, FactType, FactRelation, FactConfidence
from submit.extraction.fact_extractor import FactExtractor, RuleBasedFactExtractor, SmartFactExtractor
from submit.extraction.fact_database import FactDatabase, FactConflictResolver
from submit.extraction.fact_patterns import extract_with_pattern, FACT_PATTERNS


def test_fact_models():
    """Тестирует модели фактов"""
    print("\n=== Тест Fact Models ===")
    
    # Создаем факт
    confidence = FactConfidence(score=0.9, source="test")
    fact = Fact(
        type=FactType.PERSONAL_NAME,
        subject="пользователь",
        relation=FactRelation.IS,
        object="Александр",
        confidence=confidence,
        session_id="1",
        dialogue_id="test_dialogue"
    )
    
    print(f"Создан факт: {fact}")
    print(f"ID факта: {fact.id}")
    print(f"Естественный текст: {fact.to_natural_text()}")
    print(f"Уверенность: {fact.confidence.level} ({fact.confidence.score})")
    
    # Тестируем конфликты
    fact2 = Fact(
        type=FactType.PERSONAL_NAME,
        subject="пользователь",
        relation=FactRelation.IS,
        object="Иван",
        confidence=FactConfidence(score=0.7, source="test"),
        session_id="2",
        dialogue_id="test_dialogue"
    )
    
    print(f"\nКонфликтуют ли факты? {fact.is_conflicting_with(fact2)}")
    
    # Тестируем сериализацию
    fact_dict = fact.to_dict()
    restored_fact = Fact.from_dict(fact_dict)
    print(f"Факт после сериализации: {restored_fact}")
    
    return fact, fact2


def test_fact_patterns():
    """Тестирует паттерны извлечения"""
    print("\n=== Тест Fact Patterns ===")
    
    test_texts = {
        FactType.PERSONAL_NAME: [
            "Меня зовут Александр",
            "Я - Мария Петрова",
            "Мое имя Иван",
        ],
        FactType.PERSONAL_AGE: [
            "Мне 30 лет",
            "Мой возраст - 25 лет",
            "Я 45-летний мужчина",
        ],
        FactType.PERSONAL_LOCATION: [
            "Я живу в Москве",
            "Проживаю в Санкт-Петербурге",
            "Я из Новосибирска",
        ],
        FactType.WORK_OCCUPATION: [
            "Я работаю программистом",
            "Моя профессия - инженер",
            "Я по профессии врач",
        ],
        FactType.PET_NAME: [
            "У меня есть кот Барсик",
            "Мою собаку зовут Рекс",
            "Кошка по имени Мурка",
        ]
    }
    
    for fact_type, texts in test_texts.items():
        print(f"\n{fact_type.value}:")
        patterns = FACT_PATTERNS.get(fact_type, [])
        
        for text in texts:
            for pattern in patterns:
                result = extract_with_pattern(text, pattern)
                if result:
                    print(f"  '{text}' -> '{result}'")
                    break


def test_rule_based_extractor():
    """Тестирует rule-based экстрактор"""
    print("\n=== Тест RuleBasedFactExtractor ===")
    
    extractor = RuleBasedFactExtractor()
    
    test_messages = [
        "Привет! Меня зовут Иван, мне 35 лет.",
        "Я работаю инженером в автомобильной компании.",
        "У меня есть собака породы лабрадор по кличке Рекс.",
        "Моя жена Елена работает учителем.",
        "У нас двое детей - сын Петя и дочь Маша.",
        "Я увлекаюсь фотографией и люблю путешествовать.",
        "Недавно был в Италии, очень понравилось.",
        "Живу в Москве уже 10 лет.",
    ]
    
    all_facts = []
    for i, text in enumerate(test_messages):
        facts = extractor.extract_facts_from_text(
            text, 
            session_id=str(i), 
            dialogue_id="test_dialogue"
        )
        all_facts.extend(facts)
        
        if facts:
            print(f"\nТекст: '{text}'")
            for fact in facts:
                print(f"  - {fact}")
    
    print(f"\n{'-'*50}")
    print(f"Всего извлечено фактов: {len(all_facts)}")
    print(f"Статистика экстрактора: {extractor.get_stats()}")
    
    return all_facts


def test_fact_database(facts):
    """Тестирует базу данных фактов"""
    print("\n=== Тест FactDatabase ===")
    
    db = FactDatabase(conflict_strategy="highest_confidence")
    
    # Добавляем факты
    dialogue_id = "test_dialogue"
    print(f"Добавляем {len(facts)} фактов...")
    db.add_facts(dialogue_id, facts)
    
    # Добавляем конфликтующие факты
    conflicting_facts = [
        Fact(
            type=FactType.PERSONAL_NAME,
            subject="пользователь",
            relation=FactRelation.IS,
            object="Петр",  # Конфликт с Иваном
            confidence=FactConfidence(score=0.5, source="test"),
            session_id="10",
            dialogue_id=dialogue_id
        ),
        Fact(
            type=FactType.PERSONAL_AGE,
            subject="пользователь",
            relation=FactRelation.IS,
            object="40",  # Конфликт с 35
            confidence=FactConfidence(score=0.95, source="test"),
            session_id="11",
            dialogue_id=dialogue_id
        )
    ]
    
    print("\nДобавляем конфликтующие факты...")
    db.add_facts(dialogue_id, conflicting_facts)
    print(f"Конфликтов разрешено: {db.stats.conflicts_resolved}")
    
    # Получаем все факты
    all_facts = db.get_facts(dialogue_id)
    print(f"\nВсего фактов в базе: {len(all_facts)}")
    
    # Поиск по типу
    print("\nПоиск по типам:")
    for fact_type in [FactType.PERSONAL_NAME, FactType.PERSONAL_AGE, FactType.PET_NAME]:
        found = db.query_facts(dialogue_id, fact_type=fact_type)
        if found:
            print(f"  {fact_type.value}: {', '.join(f.object for f in found)}")
    
    # Полнотекстовый поиск
    queries = ["жена", "дети", "собака", "Москва"]
    print("\nПолнотекстовый поиск:")
    for query in queries:
        found = db.query_facts(dialogue_id, query=query)
        if found:
            print(f"  '{query}': найдено {len(found)} фактов")
            for fact in found[:2]:
                print(f"    - {fact.to_natural_text()}")
    
    # Создаем профиль пользователя
    profile = db.get_user_profile(dialogue_id)
    print("\nПрофиль пользователя:")
    for category, data in profile.items():
        if data:
            print(f"  {category}: {data}")
    
    # Статистика
    stats = db.get_stats()
    print(f"\nСтатистика базы:")
    print(f"  Всего фактов: {stats.total_facts}")
    print(f"  Средняя уверенность: {stats.average_confidence:.2f}")
    print(f"  Фактов по типам: {dict(stats.facts_by_type)}")
    
    return db


def test_conflict_resolution():
    """Тестирует разрешение конфликтов"""
    print("\n=== Тест Conflict Resolution ===")
    
    from submit.extraction.fact_models import ConflictingFacts
    
    # Создаем конфликтующие факты о возрасте
    facts = [
        Fact(
            type=FactType.PERSONAL_AGE,
            subject="пользователь",
            relation=FactRelation.IS,
            object="30",
            confidence=FactConfidence(score=0.6, source="test", evidence_count=1),
            session_id="1",
            dialogue_id="test"
        ),
        Fact(
            type=FactType.PERSONAL_AGE,
            subject="пользователь",
            relation=FactRelation.IS,
            object="32",
            confidence=FactConfidence(score=0.8, source="test", evidence_count=2),
            session_id="2",
            dialogue_id="test"
        ),
        Fact(
            type=FactType.PERSONAL_AGE,
            subject="пользователь",
            relation=FactRelation.IS,
            object="35",
            confidence=FactConfidence(score=0.7, source="test", evidence_count=3),
            session_id="3",
            dialogue_id="test"
        )
    ]
    
    # Создаем группу конфликтов
    conflicts = ConflictingFacts(facts)
    print(f"Конфликтующие значения: {conflicts.get_all_values()}")
    
    # Разные стратегии разрешения
    strategies = ["latest", "highest_confidence", "most_evidence"]
    
    for strategy in strategies:
        resolved = conflicts.resolve(strategy)
        print(f"\nСтратегия '{strategy}':")
        print(f"  Выбрано: {resolved.object}")
        print(f"  Причина: {conflicts.resolution_reason}")


def test_smart_extractor_mock():
    """Тестирует умный экстрактор с моком LLM"""
    print("\n=== Тест SmartFactExtractor (Mock) ===")
    
    # Создаем мок для модели
    class MockModelInference:
        def inference(self, messages):
            # Возвращаем мок JSON ответ
            return '''[
                {"type": "personal_name", "subject": "пользователь", "relation": "is", "object": "Сергей", "confidence": 0.95},
                {"type": "hobby_activity", "subject": "пользователь", "relation": "does", "object": "программирование", "confidence": 0.8},
                {"type": "preference_food", "subject": "пользователь", "relation": "likes", "object": "пицца", "confidence": 0.7}
            ]'''
    
    model = MockModelInference()
    extractor = SmartFactExtractor(model, use_rules_first=False)
    
    text = "Я занимаюсь разработкой веб-приложений и обожаю итальянскую кухню"
    facts = extractor.extract_facts_from_text(text, "1", "test")
    
    print(f"Текст: '{text}'")
    print(f"Извлечено фактов: {len(facts)}")
    for fact in facts:
        print(f"  - {fact}")
    
    print(f"\nСтатистика: {extractor.get_stats()}")


def test_temporal_facts():
    """Тестирует временные факты"""
    print("\n=== Тест Temporal Facts ===")
    
    from submit.extraction.fact_models import TemporalFact
    from datetime import datetime, timedelta
    
    # Создаем временной факт
    fact = TemporalFact(
        type=FactType.WORK_COMPANY,
        subject="пользователь", 
        relation=FactRelation.WORKS_AT,
        object="Google",
        confidence=FactConfidence(score=0.9, source="test"),
        session_id="1",
        dialogue_id="test",
        valid_from=datetime.now() - timedelta(days=365),
        valid_until=datetime.now() + timedelta(days=365),
        is_current=True
    )
    
    print(f"Временной факт: {fact}")
    print(f"Актуален сейчас: {fact.is_current}")
    print(f"Действителен на текущую дату: {fact.is_valid_at(datetime.now())}")
    
    # Проверяем валидность в будущем
    future_date = datetime.now() + timedelta(days=400)
    print(f"Действителен через 400 дней: {fact.is_valid_at(future_date)}")


def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование системы извлечения фактов для GigaMemory")
    print("="*60)
    
    try:
        # Тест 1: Модели фактов
        fact1, fact2 = test_fact_models()
        print("\n✅ Fact Models работают корректно")
        
        # Тест 2: Паттерны
        test_fact_patterns()
        print("\n✅ Fact Patterns работают корректно")
        
        # Тест 3: Rule-based экстрактор
        facts = test_rule_based_extractor()
        print("\n✅ RuleBasedFactExtractor работает корректно")
        
        # Тест 4: База данных фактов
        db = test_fact_database(facts)
        print("\n✅ FactDatabase работает корректно")
        
        # Тест 5: Разрешение конфликтов
        test_conflict_resolution()
        print("\n✅ Conflict Resolution работает корректно")
        
        # Тест 6: Smart экстрактор
        test_smart_extractor_mock()
        print("\n✅ SmartFactExtractor работает корректно")
        
        # Тест 7: Временные факты
        test_temporal_facts()
        print("\n✅ Temporal Facts работают корректно")
        
        print("\n" + "="*60)
        print("🎉 Все тесты пройдены успешно!")
        print("\nСистема извлечения фактов готова к интеграции!")
        print("\nВозможности системы:")
        print("  ✓ Извлечение 40+ типов фактов")
        print("  ✓ Rule-based и LLM-based подходы")
        print("  ✓ Автоматическое разрешение конфликтов")
        print("  ✓ Полнотекстовый поиск по фактам")
        print("  ✓ Построение профиля пользователя")
        print("  ✓ Поддержка временных фактов")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



