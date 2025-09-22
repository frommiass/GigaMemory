#!/usr/bin/env python3
"""
Тест для проверки правильности нумерации сессий
"""
import sys
import os

# Добавляем пути для mock-модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'start'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импортируем mock-модули
import mock_vllm
import mock_transformers

# Заменяем оригинальные модули на mock-версии
sys.modules['vllm'] = mock_vllm
sys.modules['transformers'] = mock_transformers

from models import Message, Session, Dialog
from submit.model_inference import SubmitModelWithMemory

def test_session_numbering():
    """Тестирует правильность нумерации сессий"""
    
    # Создаем тестовый диалог с сессиями, как в исходных данных
    sessions = [
        Session(
            id="1",
            messages=[
                Message(role="user", content="Меня зовут Иван", session_id="1"),
                Message(role="assistant", content="Привет, Иван!", session_id="1")
            ]
        ),
        Session(
            id="2", 
            messages=[
                Message(role="user", content="У меня есть кот", session_id="2"),
                Message(role="assistant", content="Как интересно!", session_id="2")
            ]
        ),
        Session(
            id="3",
            messages=[
                Message(role="user", content="Моя собака зовется Лайка", session_id="3"),
                Message(role="assistant", content="Красивое имя!", session_id="3")
            ]
        )
    ]
    
    dialog = Dialog(id="test_dialog", sessions=sessions, question="Как меня зовут?")
    
    # Создаем модель (используем mock для тестирования)
    model = SubmitModelWithMemory("mock_model")
    
    # Обрабатываем сообщения
    messages = dialog.get_messages()
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            message_pair = [messages[i], messages[i + 1]]
            model.write_to_memory(message_pair, dialog.id)
    
    # Получаем память и проверяем нумерацию
    memory = model.storage.get_memory(dialog.id)
    
    print("Проверка нумерации сессий:")
    print("=" * 50)
    
    for msg in memory:
        if msg.role == "user":
            print(f"Сообщение: {msg.content}")
            print(f"Session ID: {msg.session_id}")
            print("-" * 30)
    
    # Проверяем, что session_id соответствуют исходным данным
    expected_sessions = ["1", "2", "3"]
    actual_sessions = [msg.session_id for msg in memory if msg.role == "user"]
    
    print(f"Ожидаемые сессии: {expected_sessions}")
    print(f"Фактические сессии: {actual_sessions}")
    
    if actual_sessions == expected_sessions:
        print("✅ Нумерация сессий работает правильно!")
        return True
    else:
        print("❌ Ошибка в нумерации сессий!")
        return False

if __name__ == "__main__":
    test_session_numbering()
