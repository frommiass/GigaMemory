#!/usr/bin/env python3
"""
Финальный тест фильтрации копипаста
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
from submit.rag.interface import RAGInterface
from submit.filters.message_cleaner import is_copy_paste_content


def test_final_copypaste():
    """Финальный тест всех случаев копипаста"""
    print("=== ФИНАЛЬНЫЙ ТЕСТ ФИЛЬТРАЦИИ КОПИПАСТА ===\n")
    
    # Тестовые сообщения с разными типами копипаста
    test_cases = [
        # Личные сообщения (должны пройти)
        ("Привет! Как дела?", False),
        ("Я работаю программистом в Яндексе", False),
        ("Мой любимый фильм - Матрица", False),
        ("Я живу в Москве уже 10 лет", False),
        ("Мне нужно сбросить 5 килограмм", False),
        
        # Явные маркеры копипаста (должны быть отфильтрованы)
        ("Помоги мне с домашним заданием", True),
        ("Переведи этот текст на английский", True),
        ("Расскажи мне о квантовой физике", True),
        ("Напиши реферат по истории", True),
        ("Объясни, как работает блокчейн", True),
        ("Сделай презентацию по биологии", True),
        ("Найди ошибки в этом коде", True),
        ("Скопировал код из интернета", True),
        ("Вот ссылка на статью: https://example.com", True),
        ("Перескажи эту книгу кратко", True),
        
        # Социальные сети
        ("Сделай рилз для Instagram", True),
        ("Создай сторис для ВКонтакте", True),
        ("Напиши пост для Facebook", True),
        
        # Длинные технические тексты
        ("Что нужно автомобилю В-класса, чтобы стать популярным? Ответ очевиден: как можно реже ломаться и радовать своего владельца минимальной стоимостью обслуживания. В своё время этим подкупал Renault Logan первого поколения, потом герои сменились – появились Hyundai Solaris и Kia Rio, которые могли порадовать ещё и приличными автоматами и даже неплохими комплектациями.", True),
    ]
    
    print("1. ТЕСТ ОПРЕДЕЛЕНИЯ КОПИПАСТА:")
    print("-" * 50)
    
    correct = 0
    total = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result = is_copy_paste_content(text)
        status = "✓" if result == expected else "✗"
        type_text = "КОПИПАСТ" if result else "ЛИЧНОЕ"
        expected_text = "КОПИПАСТ" if expected else "ЛИЧНОЕ"
        
        print(f"{i:2d}. {status} [{type_text:>8}] {text[:60]}{'...' if len(text) > 60 else ''}")
        if result != expected:
            print(f"    Ожидалось: {expected_text}, получено: {type_text}")
        else:
            correct += 1
    
    print(f"\nРезультат: {correct}/{total} правильных определений ({correct/total*100:.1f}%)")
    
    # Тест RAG системы
    print("\n2. ТЕСТ RAG СИСТЕМЫ:")
    print("-" * 50)
    
    # Создаем сообщения с копипастом
    messages = [
        Message(role="user", content="Привет! Меня зовут Иван"),
        Message(role="user", content="Я работаю программистом в Яндексе"),
        Message(role="user", content="Помоги мне с домашним заданием по математике"),  # копипаст
        Message(role="user", content="Мой любимый язык программирования - Python"),
        Message(role="user", content="Переведи этот текст на английский"),  # копипаст
        Message(role="user", content="Я живу в Москве уже 10 лет"),
        Message(role="user", content="Сделай рилз для Instagram"),  # копипаст
        Message(role="assistant", content="Конечно, помогу!"),
    ]
    
    # Создаем RAG интерфейс
    rag = RAGInterface()
    
    # Тестируем вопрос
    question = "Где ты работаешь?"
    
    print(f"Вопрос: {question}")
    print("\nИсходные сообщения:")
    for i, msg in enumerate(messages, 1):
        if msg.role == "user":
            is_copypaste = is_copy_paste_content(msg.content)
            status = " [КОПИПАСТ]" if is_copypaste else ""
            print(f"{i:2d}. [{msg.role:>10}]{status} {msg.content}")
        else:
            print(f"{i:2d}. [{msg.role:>10}] {msg.content}")
    
    # Получаем ответ
    try:
        answer = rag.answer_question(question, "test_dialogue", messages)
        print(f"\nОтвет RAG системы:")
        print(answer[:200] + "..." if len(answer) > 200 else answer)
        
        # Получаем оценки сессий
        scores = rag.get_all_session_scores(question, "test_dialogue", messages)
        print(f"\nАнализ сессий:")
        print(f"Всего сессий: {scores.get('total_sessions', 0)}")
        print(f"Релевантных сессий: {scores.get('relevant_sessions', 0)}")
        print(f"Стратегия: {scores.get('strategy', 'unknown')}")
        
        # Проверяем, что копипаст не попал в промпт
        if "домашним заданием" not in answer and "переведи" not in answer and "рилз" not in answer:
            print("✓ Копипаст успешно отфильтрован из промпта")
        else:
            print("✗ Копипаст попал в промпт (проблема!)")
        
    except Exception as e:
        print(f"Ошибка при тестировании RAG: {e}")
    
    print("\n=== ТЕСТ ЗАВЕРШЕН ===")


if __name__ == "__main__":
    test_final_copypaste()



