#!/usr/bin/env python3
"""
Скрипт для тестирования новой архитектуры SmartMemory на 4 диалогах
Читает данные, индексирует через OptimizedSmartMemory, сохраняет промпты
"""
import os
import sys
import json
from pathlib import Path

# Добавляем src в PYTHONPATH для импорта модулей проекта
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Создаем моки для внешних библиотек (если доступны)
try:
    import mock_transformers
    import mock_vllm
    sys.modules['transformers'] = mock_transformers
    sys.modules['vllm'] = mock_vllm
except ImportError:
    pass

# Импорты проекта
from models import Dialog, Message
from submit.core.data_loader import DataLoader
from submit.smart_memory_optimized import OptimizedSmartMemory


def load_dialogs(data_file: str) -> list[Dialog]:
    """Загружает диалоги из JSONL файла с фильтрацией копипаста"""
    data_loader = DataLoader()
    dialogs, analysis = data_loader.load_dialogs_with_analysis(data_file, apply_filtering=True)

    print(f"Загружено диалогов: {analysis['total_dialogs']} -> {analysis['filtered_dialogs']}")
    print(f"Загружено сессий: {analysis['total_sessions']} -> {analysis['filtered_sessions']}")
    print(f"Загружено сообщений: {analysis['total_messages']} -> {analysis['filtered_messages']}")
    print(f"Удалено диалогов: {analysis['dialogs_removed']}")
    print(f"Удалено сессий: {analysis['sessions_removed']}")
    print(f"Удалено сообщений: {analysis['messages_removed']}")

    return dialogs


def build_prompt_with_smart(memory: OptimizedSmartMemory, dialog: Dialog) -> tuple[str, dict]:
    """Строит промпт через новую архитектуру и возвращает также анализ сессий"""
    # Собираем все сообщения
    all_messages: list[Message] = []
    for session in dialog.sessions:
        all_messages.extend(session.messages)

    # Индексируем/обрабатываем диалог в новой системе
    memory.process_dialogue_optimized(str(dialog.id), all_messages)

    # Получаем промпт напрямую из движка (как в новой архитектуре)
    prompt, metadata = memory.rag_engine.process_question(dialog.question, str(dialog.id), all_messages)

    # Для анализа релевантности используем ту же логику, что и раньше
    session_analysis = memory.rag_engine.get_all_session_scores(dialog.question, str(dialog.id), all_messages)

    return prompt, session_analysis


def save_prompt(dialog_id: int, question: str, prompt: str, output_dir: str, suffix: str = "smart"):
    """Сохраняет промпт в файл"""
    filename = f"prompt_{suffix}_{dialog_id}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Сохранен промпт: {filename}")


def save_session_analysis(dialog_id: int, question: str, session_analysis: dict, output_dir: str, suffix: str = "smart"):
    """Сохраняет анализ релевантности сессий в JSON файл"""
    filename = f"session_analysis_{suffix}_{dialog_id}.json"
    filepath = os.path.join(output_dir, filename)

    # Поддерживаем тот же формат, что и в исходном скрипте
    all_scores = session_analysis.get('all_scores', {})
    total_sessions = session_analysis.get('total_sessions', 0)
    relevant_sessions = session_analysis.get('relevant_sessions', 0)

    relevance_scores = {}
    for session_id, info in all_scores.items():
        relevance_scores[session_id] = {
            'score': info.get('relevance_score', 0.0),
            'status': info.get('status'),
            'matched_keywords': info.get('matched_keywords', []),
            'content_length': info.get('content_length', 0),
            'message_count': info.get('message_count', 0)
        }

    # Отсортированные сессии
    sorted_sessions = session_analysis.get('sorted_sessions', [])

    analysis_data = {
        "dialog_id": dialog_id,
        "question": question,
        "analysis": {
            "question": question,
            "topic": session_analysis.get('topic'),
            "confidence": session_analysis.get('confidence', 0),
            "strategy": session_analysis.get('strategy', 'unknown'),
            "total_sessions": total_sessions,
            "relevance_scores": relevance_scores,
            "sorted_sessions": sorted_sessions
        },
        "summary": {
            "total_sessions": total_sessions,
            "relevant_sessions": relevant_sessions,
            "topic": session_analysis.get('topic'),
            "confidence": session_analysis.get('confidence', 0),
            "strategy": session_analysis.get('strategy', 'unknown')
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)

    print(f"Сохранен анализ сессий: {filename}")


def main():
    data_file = "data/format_example.jsonl"
    output_dir = "prompts"

    os.makedirs(output_dir, exist_ok=True)

    print("Загружаем диалоги...")
    dialogs = load_dialogs(data_file)
    print(f"Загружено {len(dialogs)} диалогов")

    # Берем первые 4 диалога (как в исходной постановке)
    dialogs_to_process = dialogs[:4]

    print("Инициализируем SmartMemory...")
    memory = OptimizedSmartMemory()

    print("Обрабатываем диалоги через новую архитектуру...")
    for i, dialog in enumerate(dialogs_to_process, 1):
        print(f"Обрабатываем диалог {i}/{len(dialogs_to_process)} (ID: {dialog.id})")
        try:
            prompt, session_analysis = build_prompt_with_smart(memory, dialog)

            print(f"  Длина промпта: {len(prompt)} символов")
            if len(prompt) < 100:
                print("  ВНИМАНИЕ: Промпт слишком короткий!")
                print(f"  Полный промпт: {prompt}")

            save_prompt(dialog.id, dialog.question, prompt, output_dir, suffix="smart")
            save_session_analysis(dialog.id, dialog.question, session_analysis, output_dir, suffix="smart")

            print(f"  Вопрос: {dialog.question}")
            print(f"  Промпт: {prompt[:150]}...")
            print()
        except Exception as e:
            print(f"  Ошибка при обработке диалога {dialog.id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("Обработка завершена!")
    print(f"Промпты и анализ сессий сохранены в папке: {output_dir}")


if __name__ == "__main__":
    main()

