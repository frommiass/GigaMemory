#!/usr/bin/env python
"""
Упрощенный тест интеграции факт-ориентированного RAG
"""
import sys
import json
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

def test_components():
    """Тестирует отдельные компоненты"""
    
    print("🧪 Тестирование компонентов факт-ориентированного RAG...")
    
    # Тестируем импорт классификатора
    try:
        from submit.questions.fact_based_classifier import FactBasedQuestionClassifier
        print("✅ FactBasedQuestionClassifier импортирован успешно")
    except Exception as e:
        print(f"❌ Ошибка импорта FactBasedQuestionClassifier: {e}")
        return False
    
    # Тестируем импорт RAG движка
    try:
        from submit.rag.fact_based_rag import FactBasedRAGEngine
        print("✅ FactBasedRAGEngine импортирован успешно")
    except Exception as e:
        print(f"❌ Ошибка импорта FactBasedRAGEngine: {e}")
        return False
    
    # Тестируем импорт моделей фактов
    try:
        from submit.extraction.fact_models import FactType, Fact, FactConfidence, FactRelation
        print("✅ Модели фактов импортированы успешно")
    except Exception as e:
        print(f"❌ Ошибка импорта моделей фактов: {e}")
        return False
    
    # Тестируем импорт базы фактов
    try:
        from submit.extraction.fact_database import FactDatabase
        print("✅ FactDatabase импортирована успешно")
    except Exception as e:
        print(f"❌ Ошибка импорта FactDatabase: {e}")
        return False
    
    # Тестируем создание классификатора
    try:
        classifier = FactBasedQuestionClassifier()
        print("✅ FactBasedQuestionClassifier создан успешно")
    except Exception as e:
        print(f"❌ Ошибка создания FactBasedQuestionClassifier: {e}")
        return False
    
    # Тестируем создание базы фактов
    try:
        fact_database = FactDatabase()
        print("✅ FactDatabase создана успешно")
    except Exception as e:
        print(f"❌ Ошибка создания FactDatabase: {e}")
        return False
    
    # Тестируем создание RAG движка
    try:
        rag_engine = FactBasedRAGEngine(fact_database)
        print("✅ FactBasedRAGEngine создан успешно")
    except Exception as e:
        print(f"❌ Ошибка создания FactBasedRAGEngine: {e}")
        return False
    
    # Тестируем классификацию вопросов
    test_questions = [
        "Каким спортом я занимаюсь?",
        "Кем я работаю?",
        "Какая порода у моей собаки?",
        "Сигареты какой марки я предпочитаю?",
        "Как меня зовут?",
        "Сколько мне лет?"
    ]
    
    print("\n🔍 Тестирование классификации вопросов:")
    for question in test_questions:
        try:
            fact_type, confidence = classifier.classify_question(question)
            print(f"  '{question}' -> {fact_type} (уверенность: {confidence:.2f})")
        except Exception as e:
            print(f"  ❌ Ошибка классификации '{question}': {e}")
            return False
    
    # Тестируем обработку вопросов
    print("\n🔍 Тестирование обработки вопросов:")
    for question in test_questions:
        try:
            prompt, metadata = rag_engine.process_question(question, "test_dialogue")
            print(f"  '{question}' -> {metadata['strategy']} (фактов: {metadata['facts_found']})")
        except Exception as e:
            print(f"  ❌ Ошибка обработки '{question}': {e}")
            return False
    
    print("\n✅ Все компоненты работают корректно!")
    return True

if __name__ == "__main__":
    success = test_components()
    sys.exit(0 if success else 1)


