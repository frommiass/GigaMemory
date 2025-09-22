#!/usr/bin/env python3
"""
Тест для проверки конкретного сообщения
"""
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from submit.core.data_loader import DataLoader
from submit.filters.message_cleaner import is_copy_paste_content


def test_specific_message():
    """Тестирует конкретное сообщение из промпта"""
    print("Тестируем конкретное сообщение из промпта...")
    
    # Загружаем данные
    data_loader = DataLoader()
    dialogs = data_loader.load_dialogs("data/format_example.jsonl", apply_filtering=True)
    
    # Ищем сообщение с копипастом про Kia Rio
    for dialog in dialogs:
        if dialog.id == 1:  # Первый диалог
            print(f"Диалог {dialog.id}: {dialog.question}")
            
            for session in dialog.sessions:
                for msg in session.messages:
                    if msg.role == "user" and "Kia Rio" in msg.content:
                        print(f"\nНайдено сообщение с Kia Rio:")
                        print(f"  Длина: {len(msg.content)} символов")
                        print(f"  Первые 200 символов: {msg.content[:200]}...")
                        
                        # Проверяем, определяется ли как копипаст
                        is_copypaste = is_copy_paste_content(msg.content)
                        print(f"  Копипаст: {is_copypaste}")
                        
                        # Проверяем части сообщения
                        if "\n\n" in msg.content:
                            parts = msg.content.split("\n\n")
                            print(f"  Частей: {len(parts)}")
                            
                            for i, part in enumerate(parts[:3]):
                                print(f"    Часть {i+1}: копипаст={is_copy_paste_content(part)}")
                        
                        return
    
    print("Сообщение с Kia Rio не найдено")


if __name__ == "__main__":
    test_specific_message()
