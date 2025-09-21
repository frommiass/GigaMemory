#!/usr/bin/env python3
"""
Скрипт для тестирования модели памяти на 4 диалогах
Читает данные, обрабатывает через SubmitModelWithMemory, сохраняет промпты
"""
import os
import sys
import json
from pathlib import Path

# Добавляем src в PYTHONPATH для импорта модулей проекта
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Создаем моки для внешних библиотек
import mock_transformers
import mock_vllm

# Подменяем импорты в sys.modules
sys.modules['transformers'] = mock_transformers
sys.modules['vllm'] = mock_vllm

# Теперь импортируем модули проекта
from models import Dialog
from submit import SubmitModelWithMemory


def load_dialogs(data_file: str) -> list[Dialog]:
    """Загружает диалоги из JSONL файла"""
    dialogs = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                dialog = Dialog.from_dict(data)
                dialogs.append(dialog)
    
    return dialogs


def process_dialog(model: SubmitModelWithMemory, dialog: Dialog) -> str:
    """Обрабатывает один диалог и возвращает промпт"""
    # Получаем все сообщения диалога
    messages = dialog.get_messages()
    
    # Обрабатываем сообщения парами (user + assistant)
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            message_pair = [messages[i], messages[i + 1]]
            model.write_to_memory(message_pair, dialog.id)
    
    # Получаем промпт (не ответ!)
    prompt = model.answer_to_question_mock(dialog.id, dialog.question)
    
    # Очищаем память
    model.clear_memory(dialog.id)
    
    return prompt


def save_prompt(dialog_id: int, question: str, prompt: str, output_dir: str):
    """Сохраняет промпт в файл"""
    filename = f"prompt_{dialog_id}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"Сохранен промпт: {filename}")


def main():
    """Основная функция"""
    # Пути к файлам
    data_file = "../data/format_example.jsonl"
    output_dir = "prompts"
    model_path = "/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16"  # Мок-путь
    
    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    print("Загружаем диалоги...")
    dialogs = load_dialogs(data_file)
    print(f"Загружено {len(dialogs)} диалогов")
    
    print("Инициализируем модель...")
    model = SubmitModelWithMemory(model_path)
    
    print("Обрабатываем диалоги...")
    for i, dialog in enumerate(dialogs, 1):
        print(f"Обрабатываем диалог {i}/{len(dialogs)} (ID: {dialog.id})")
        
        try:
            prompt = process_dialog(model, dialog)
            save_prompt(dialog.id, dialog.question, prompt, output_dir)
            print(f"  Вопрос: {dialog.question}")
            print(f"  Промпт: {prompt[:100]}...")
            print()
            
        except Exception as e:
            print(f"  Ошибка при обработке диалога {dialog.id}: {e}")
            continue
    
    print("Обработка завершена!")


if __name__ == "__main__":
    main()
