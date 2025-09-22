#!/usr/bin/env python3
"""
Простой тест интеграции RAG компонентов
"""
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
from submit.rag import RAGInterface, rag_interface


def test_rag_integration():
    """Тестирует интеграцию RAG компонентов"""
    print("🧪 Тестирование RAG интеграции...")
    
    # Создаем тестовые сообщения
    test_messages = [
        Message(role="user", content="Привет, меня зовут Иван", session_id="1"),
        Message(role="assistant", content="Привет, Иван! Рад познакомиться!"),
        Message(role="user", content="У меня есть кот по имени Барсик", session_id="1"),
        Message(role="assistant", content="Как интересно! Расскажи о своем коте."),
        Message(role="user", content="Мой кот любит играть с мячиком", session_id="2"),
        Message(role="assistant", content="Звучит забавно!"),
        Message(role="user", content="Я работаю программистом в IT компании", session_id="3"),
        Message(role="assistant", content="Отлично! Какие технологии используешь?"),
    ]
    
    # Создаем RAG интерфейс
    rag = RAGInterface()
    
    # Тест 1: Классификация вопроса
    print("\n1️⃣ Тестирование классификации вопросов:")
    test_questions = [
        "Как меня зовут?",
        "Какие у меня животные?",
        "Где я работаю?",
        "Что я делаю в свободное время?"
    ]
    
    for question in test_questions:
        topic, confidence = rag.classify_question(question)
        print(f"   Вопрос: '{question}'")
        print(f"   Тема: {topic}, Уверенность: {confidence:.2f}")
    
    # Тест 2: Валидация вопросов
    print("\n2️⃣ Тестирование валидации вопросов:")
    validation_tests = [
        "Как меня зовут?",
        "",
        "   ",
        "Очень длинный вопрос с множеством слов для тестирования классификации и обработки"
    ]
    
    for question in validation_tests:
        validation = rag.validate_question(question)
        print(f"   Вопрос: '{question[:30]}...'")
        print(f"   Валидный: {validation['valid']}, Стратегия: {validation.get('strategy', 'N/A')}")
    
    # Тест 3: Группировка сессий
    print("\n3️⃣ Тестирование группировки сессий:")
    sessions = rag.rag_engine.session_grouper.group_messages_by_sessions(test_messages, "test_dialogue")
    print(f"   Количество сессий: {len(sessions)}")
    for session_id, messages in sessions.items():
        print(f"   Сессия {session_id}: {len(messages)} сообщений")
    
    # Тест 4: Поиск релевантных сессий
    print("\n4️⃣ Тестирование поиска релевантных сессий:")
    question = "Какие у меня животные?"
    relevant_sessions = rag.get_relevant_sessions(question, "test_dialogue", test_messages)
    print(f"   Вопрос: '{question}'")
    print(f"   Релевантных сессий: {len(relevant_sessions)}")
    
    # Тест 5: Ранжирование сессий
    print("\n5️⃣ Тестирование ранжирования сессий:")
    ranking = rag.get_session_ranking(question, "test_dialogue", test_messages)
    print(f"   Ранжированных сессий: {len(ranking)}")
    for session_id, score in ranking[:3]:  # Топ-3
        print(f"   Сессия {session_id}: {score:.3f}")
    
    # Тест 6: Генерация ответа
    print("\n6️⃣ Тестирование генерации ответа:")
    answer = rag.answer_question(question, "test_dialogue", test_messages)
    print(f"   Вопрос: '{question}'")
    print(f"   Ответ (промпт): {answer[:200]}...")
    
    # Тест 7: Анализ вопроса
    print("\n7️⃣ Тестирование анализа вопроса:")
    analysis = rag.get_question_context(question, "test_dialogue", test_messages)
    print(f"   Стратегия: {analysis.get('strategy', 'N/A')}")
    print(f"   Тема: {analysis.get('topic', 'N/A')}")
    print(f"   Уверенность: {analysis.get('confidence', 0):.2f}")
    print(f"   Всего сессий: {analysis.get('total_sessions', 0)}")
    
    # Тест 8: Статистика системы
    print("\n8️⃣ Статистика системы:")
    stats = rag.get_system_stats()
    print(f"   Доступных тем: {len(stats.get('available_topics', []))}")
    print(f"   Порог уверенности: {stats['config']['confidence_threshold']}")
    print(f"   Максимум сессий: {stats['config']['max_relevant_sessions']}")
    
    print("\n✅ Все тесты завершены успешно!")


if __name__ == "__main__":
    test_rag_integration()
