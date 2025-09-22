#!/usr/bin/env python3
"""
Тест классификации вопросов
"""
import sys
import os

# Добавляем пути для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импортируем только нужные модули, избегая проблем с vllm
from submit.questions.topics import get_all_topics
from submit.questions.confidence import get_confidence_level


def test_classification():
    """Тестирует классификацию вопросов"""
    
    # Создаем простой классификатор без импорта всего модуля
    from submit.questions.classifier import QuestionClassifier
    classifier = QuestionClassifier()
    
    # Тестовые вопросы
    test_questions = [
        "Каким спортом я занимаюсь?",
        "Кем я работаю?",
        "Какая порода у моей собаки?",
        "Сигареты какой марки я предпочитаю?",
        "Как меня зовут?",
        "Где я живу?",
        "Какие у меня хобби?",
        "Куда я ездил в отпуск?",
        "Какие у меня проблемы со здоровьем?",
        "Что я люблю готовить?",
        "Как зовут моего кота?",
        "В каком магазине я покупаю одежду?",
        "В каком университете я учился?",
        "Как дела на работе?",
        "Что нового в семье?"
    ]
    
    print("Тестирование классификации вопросов:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        topic, confidence = classifier.classify_question(question)
        confidence_level = get_confidence_level(confidence)
        
        print(f"{i:2d}. Вопрос: {question}")
        print(f"    Тема: {topic or 'Не определена'}")
        print(f"    Уверенность: {confidence:.3f} ({confidence_level})")
        
        # Показываем топ-3 темы
        top_topics = classifier.get_top_topics(question, top_k=3)
        if top_topics:
            print(f"    Топ темы: {', '.join([f'{t}({c:.2f})' for t, c in top_topics])}")
        
        print("-" * 60)
    
    print("\nДоступные темы:")
    topics = classifier.get_available_topics()
    for i, topic in enumerate(topics, 1):
        print(f"{i:2d}. {topic}")


if __name__ == "__main__":
    test_classification()
