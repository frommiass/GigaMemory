#!/usr/bin/env python3
"""
Тест для проверки фильтрации в DataLoader
"""
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from submit.core.data_loader import DataLoader


def test_dataloader_filtering():
    """Тестирует фильтрацию в DataLoader"""
    print("Тестируем фильтрацию в DataLoader...")
    
    # Загружаем данные с фильтрацией
    data_loader = DataLoader()
    dialogs, analysis = data_loader.load_dialogs_with_analysis("data/format_example.jsonl", apply_filtering=True)
    
    print(f"Анализ фильтрации: {analysis}")
    
    # Проверяем первый диалог
    if dialogs:
        dialog = dialogs[0]
        print(f"\nДиалог {dialog.id}: {dialog.question}")
        print(f"Сессий: {len(dialog.sessions)}")
        
        # Проверяем сообщения в первой сессии
        if dialog.sessions:
            session = dialog.sessions[0]
            print(f"\nПервая сессия: {len(session.messages)} сообщений")
            
            for i, msg in enumerate(session.messages[:5]):  # Первые 5 сообщений
                print(f"\nСообщение {i+1}:")
                print(f"  Роль: {msg.role}")
                print(f"  Длина: {len(msg.content)} символов")
                print(f"  Первые 100 символов: {msg.content[:100]}...")
                
                # Проверяем, является ли копипастом
                from submit.filters.message_cleaner import is_copy_paste_content
                is_copypaste = is_copy_paste_content(msg.content)
                print(f"  Копипаст: {is_copypaste}")


if __name__ == "__main__":
    test_dataloader_filtering()
