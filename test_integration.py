#!/usr/bin/env python
"""
Тестовый скрипт для проверки интеграции факт-ориентированного RAG
"""
import sys
import json
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from submit.smart_memory import SmartMemory, SmartMemoryConfig
from models import Message

def test_integration():
    """Тестирует интеграцию факт-ориентированного RAG"""
    
    print("🧪 Тестирование интеграции факт-ориентированного RAG...")
    
    # Создаем конфигурацию с включенным факт-ориентированным RAG
    config = SmartMemoryConfig()
    config.use_fact_based_rag = True
    config.use_fact_extraction = True
    config.use_vector_search = False  # Отключаем для простоты тестирования
    config.use_compression = False
    
    # Инициализируем SmartMemory
    try:
        smart_memory = SmartMemory("dummy_model_path", config)
        print("✅ SmartMemory инициализирована успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False
    
    # Проверяем наличие факт-ориентированного RAG движка
    if hasattr(smart_memory, 'fact_rag_engine') and smart_memory.fact_rag_engine:
        print("✅ FactBasedRAGEngine инициализирован")
    else:
        print("❌ FactBasedRAGEngine не инициализирован")
        return False
    
    # Проверяем наличие классификатора вопросов
    if hasattr(smart_memory.fact_rag_engine, 'classifier'):
        print("✅ FactBasedQuestionClassifier инициализирован")
    else:
        print("❌ FactBasedQuestionClassifier не инициализирован")
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
        fact_type, confidence = smart_memory.fact_rag_engine.classifier.classify_question(question)
        print(f"  '{question}' -> {fact_type} (уверенность: {confidence:.2f})")
    
    print("\n✅ Интеграция работает корректно!")
    return True

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)

