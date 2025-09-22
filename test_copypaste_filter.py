#!/usr/bin/env python3
"""
Тест фильтрации копипаста в RAG системе
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
from submit.rag.interface import RAGInterface
from submit.filters.message_cleaner import is_copy_paste_content, clean_messages


def test_copypaste_detection():
    """Тестирует определение копипаста"""
    print("=== Тест определения копипаста ===")
    
    # Тестовые сообщения
    test_messages = [
        "Привет! Как дела?",
        "Мне нужно перевести этот текст на английский",
        "Я работаю программистом в IT компании",
        "Помоги мне с домашним заданием по математике",
        "Скопировал код из интернета, но он не работает",
        "Мой любимый фильм - это 'Матрица'",
        "Вот ссылка на статью: https://example.com",
        "Перескажи мне эту книгу кратко",
        "Я живу в Москве уже 5 лет",
        "Найди ошибки в этом коде: def hello(): print('world')"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        is_copypaste = is_copy_paste_content(msg)
        status = "КОПИПАСТ" if is_copypaste else "ЛИЧНОЕ"
        print(f"{i:2d}. [{status}] {msg}")
    
    print()


def test_message_filtering():
    """Тестирует фильтрацию сообщений"""
    print("=== Тест фильтрации сообщений ===")
    
    # Создаем тестовые сообщения
    messages = [
        Message(role="user", content="Привет! Как дела?"),
        Message(role="user", content="Мне нужно перевести этот текст"),
        Message(role="user", content="Я работаю программистом"),
        Message(role="user", content="Скопировал код из интернета"),
        Message(role="user", content="Мой любимый фильм - Матрица"),
        Message(role="user", content="Помоги с домашним заданием"),
        Message(role="user", content="Я живу в Москве"),
        Message(role="assistant", content="Хорошо, помогу!"),
    ]
    
    print("Исходные сообщения:")
    for i, msg in enumerate(messages, 1):
        print(f"{i:2d}. [{msg.role:>10}] {msg.content}")
    
    # Фильтруем сообщения
    filtered = clean_messages(messages)
    
    print("\nОтфильтрованные сообщения:")
    for i, msg in enumerate(filtered, 1):
        print(f"{i:2d}. [{msg.role:>10}] {msg.content}")
    
    print(f"\nИсходно: {len(messages)} сообщений")
    print(f"После фильтрации: {len(filtered)} сообщений")
    print()


def test_rag_with_copypaste():
    """Тестирует RAG систему с копипастом"""
    print("=== Тест RAG системы с копипастом ===")
    
    # Создаем тестовые сообщения с копипастом
    messages = [
        Message(role="user", content="Привет! Меня зовут Иван"),
        Message(role="user", content="Я работаю программистом в Яндексе"),
        Message(role="user", content="Скопировал код из интернета для работы"),
        Message(role="user", content="Мой любимый язык программирования - Python"),
        Message(role="user", content="Помоги мне с домашним заданием по математике"),
        Message(role="user", content="Я живу в Москве уже 10 лет"),
        Message(role="user", content="Переведи этот текст на английский язык"),
        Message(role="assistant", content="Конечно, помогу!"),
    ]
    
    # Создаем RAG интерфейс
    rag = RAGInterface()
    
    # Тестируем вопрос
    question = "Где ты работаешь?"
    
    print(f"Вопрос: {question}")
    print("\nИсходные сообщения:")
    for i, msg in enumerate(messages, 1):
        is_copypaste = is_copy_paste_content(msg.content) if msg.role == "user" else False
        status = " [КОПИПАСТ]" if is_copypaste else ""
        print(f"{i:2d}. [{msg.role:>10}]{status} {msg.content}")
    
    # Получаем ответ
    try:
        answer = rag.answer_question(question, "test_dialogue", messages)
        print(f"\nОтвет RAG системы:")
        print(answer)
        
        # Получаем оценки сессий
        scores = rag.get_all_session_scores(question, "test_dialogue", messages)
        print(f"\nАнализ сессий:")
        print(f"Всего сессий: {scores.get('total_sessions', 0)}")
        print(f"Релевантных сессий: {scores.get('relevant_sessions', 0)}")
        print(f"Стратегия: {scores.get('strategy', 'unknown')}")
        
    except Exception as e:
        print(f"Ошибка при тестировании RAG: {e}")
    
    print()


if __name__ == "__main__":
    print("Тестирование фильтрации копипаста в RAG системе\n")
    
    test_copypaste_detection()
    test_message_filtering()
    test_rag_with_copypaste()
    
    print("Тестирование завершено!")
