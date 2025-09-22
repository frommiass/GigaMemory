#!/usr/bin/env python3
"""
Тест для отладки фильтрации
"""
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from models import Message
from submit.filters.message_cleaner import is_copy_paste_content
from submit.filters.session_grouper import extract_session_content


def test_filtering_debug():
    """Тестирует фильтрацию для отладки"""
    print("Тестируем фильтрацию для отладки...")
    
    # Создаем тестовые сообщения
    messages = [
        Message(role="user", content="Привет! Как дела?"),  # Личное сообщение
        Message(role="user", content="напиши коротенько, о чём этот текст: Что нужно автомобилю В-класса, чтобы стать популярным? Ответ очевиден: как можно реже ломаться и радовать своего владельца минимальной стоимостью обслуживания. В своё время этим подкупал Renault Logan первого поколения, потом герои сменились – появились Hyundai Solaris и Kia Rio, которые могли порадовать ещё и приличными автоматами и даже неплохими комплектациями. Правда, многие считают, что эти автомобили одноразовые и сделаны без особого запаса прочности. Так ли это на самом деле? Может быть, просто надо их лучше обслуживать? И с чем можно столкнуться при эксплуатации подержанного бюджетного «корейца»? Давайте возьмём Киа Рио третьего поколения и постараемся ответить на все поставленные вопросы."),  # Копипаст
        Message(role="user", content="Я хочу узнать о машинном обучении"),  # Копипаст
        Message(role="user", content="Расскажи мне о себе"),  # Копипаст
    ]
    
    print("Исходные сообщения:")
    for i, msg in enumerate(messages):
        is_copypaste = is_copy_paste_content(msg.content)
        print(f"  {i+1}. [{msg.role}] {msg.content[:100]}...")
        print(f"     Копипаст: {is_copypaste}")
        print()
    
    # Тестируем extract_session_content
    print("Результат extract_session_content:")
    session_content = extract_session_content(messages)
    print(f"Содержимое сессии: {session_content}")
    print(f"Длина: {len(session_content)} символов")
    
    # Проверим каждое сообщение отдельно
    print("\nПроверка каждого сообщения в extract_session_content:")
    for i, msg in enumerate(messages):
        if msg.role == "user" and msg.content.strip():
            is_copypaste = is_copy_paste_content(msg.content)
            print(f"  Сообщение {i+1}: копипаст={is_copypaste}, будет включено={not is_copypaste}")


if __name__ == "__main__":
    test_filtering_debug()
