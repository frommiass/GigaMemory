#!/usr/bin/env python3
"""
Скрипт для тестирования RAG системы на 4 диалогах
Читает данные, обрабатывает через RAG систему, сохраняет промпты
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
from models import Dialog, Message
from submit.rag import RAGInterface


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


def process_dialog(rag: RAGInterface, dialog: Dialog) -> str:
    """Обрабатывает один диалог через RAG систему и возвращает промпт"""
    # Собираем все сообщения из всех сессий
    all_messages = []
    for session in dialog.sessions:
        all_messages.extend(session.messages)
    
    # Получаем промпт через RAG систему
    prompt = rag.answer_question(dialog.question, str(dialog.id), all_messages)
    
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
    
    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    print("Загружаем диалоги...")
    dialogs = load_dialogs(data_file)
    print(f"Загружено {len(dialogs)} диалогов")
    
    print("Инициализируем RAG систему...")
    rag = RAGInterface()
    
    print("Обрабатываем диалоги через RAG систему...")
    for i, dialog in enumerate(dialogs, 1):
        print(f"Обрабатываем диалог {i}/{len(dialogs)} (ID: {dialog.id})")
        
        try:
            # Получаем анализ вопроса
            validation = rag.validate_question(dialog.question)
            print(f"  Тема: {validation.get('topic', 'None')}")
            print(f"  Уверенность: {validation.get('confidence', 0):.2f}")
            print(f"  Стратегия: {validation.get('strategy', 'unknown')}")
            
            # Обрабатываем диалог
            prompt = process_dialog(rag, dialog)
            
            # Отладочная информация
            print(f"  Длина промпта: {len(prompt)} символов")
            if len(prompt) < 100:
                print(f"  ВНИМАНИЕ: Промпт слишком короткий!")
                print(f"  Полный промпт: {prompt}")
            
            save_prompt(dialog.id, dialog.question, prompt, output_dir)
            print(f"  Вопрос: {dialog.question}")
            print(f"  Промпт: {prompt[:150]}...")
            print()
            
        except Exception as e:
            print(f"  Ошибка при обработке диалога {dialog.id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Обработка завершена!")
    print(f"Промпты сохранены в папке: {output_dir}")


if __name__ == "__main__":
    main()
