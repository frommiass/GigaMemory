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

# Создаем моки для внешних библиотек (только если модули существуют)
try:
    import mock_transformers
    import mock_vllm
    # Подменяем импорты в sys.modules
    sys.modules['transformers'] = mock_transformers
    sys.modules['vllm'] = mock_vllm
except ImportError:
    pass  # Модули не найдены, продолжаем без подмены

# Теперь импортируем модули проекта
from models import Dialog, Message
from submit.rag import RAGInterface
from submit.core.data_loader import DataLoader


def load_dialogs(data_file: str) -> list[Dialog]:
    """Загружает диалоги из JSONL файла с фильтрацией копипаста"""
    # Используем центральный DataLoader с автоматической фильтрацией
    data_loader = DataLoader()
    dialogs, analysis = data_loader.load_dialogs_with_analysis(data_file, apply_filtering=True)
    
    # Выводим статистику фильтрации
    print(f"Загружено диалогов: {analysis['total_dialogs']} -> {analysis['filtered_dialogs']}")
    print(f"Загружено сессий: {analysis['total_sessions']} -> {analysis['filtered_sessions']}")
    print(f"Загружено сообщений: {analysis['total_messages']} -> {analysis['filtered_messages']}")
    print(f"Удалено диалогов: {analysis['dialogs_removed']}")
    print(f"Удалено сессий: {analysis['sessions_removed']}")
    print(f"Удалено сообщений: {analysis['messages_removed']}")
    
    return dialogs


def process_dialog(rag: RAGInterface, dialog: Dialog) -> tuple[str, dict]:
    """Обрабатывает один диалог через RAG систему и возвращает промпт и анализ сессий"""
    # Собираем все сообщения из всех сессий
    all_messages = []
    for session in dialog.sessions:
        all_messages.extend(session.messages)
    
    # Получаем промпт через RAG систему
    prompt = rag.answer_question(dialog.question, str(dialog.id), all_messages)
    
    # Получаем полные оценки всех сессий
    session_analysis = rag.get_all_session_scores(dialog.question, str(dialog.id), all_messages)
    
    return prompt, session_analysis


def save_prompt(dialog_id: int, question: str, prompt: str, output_dir: str):
    """Сохраняет промпт в файл"""
    filename = f"prompt_{dialog_id}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"Сохранен промпт: {filename}")


def save_session_analysis(dialog_id: int, question: str, session_analysis: dict, output_dir: str):
    """Сохраняет анализ релевантности сессий в JSON файл"""
    filename = f"session_analysis_{dialog_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Получаем данные из нового формата
    all_scores = session_analysis.get('all_scores', {})
    total_sessions = session_analysis.get('total_sessions', 0)
    relevant_sessions = session_analysis.get('relevant_sessions', 0)
    
    # Создаем упрощенный формат для совместимости с дополнительной информацией
    relevance_scores = {}
    for session_id, info in all_scores.items():
        relevance_scores[session_id] = {
            'score': info['relevance_score'],
            'status': info['status'],
            'matched_keywords': info['matched_keywords'],
            'content_length': info['content_length'],
            'message_count': info['message_count']
        }
    
    # Сортируем сессии по релевантности
    ranked_sessions = sorted(all_scores.items(), key=lambda x: x[1]['relevance_score'], reverse=True)
    ranked_sessions = [[int(session_id), info['relevance_score']] for session_id, info in ranked_sessions]
    
    # Получаем отсортированные сессии
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
    
    # Выводим краткую сводку
    summary = analysis_data["summary"]
    print(f"  Всего сессий: {summary['total_sessions']}")
    print(f"  Релевантных сессий: {summary['relevant_sessions']}")
    print(f"  Тема: {summary['topic'] or 'Не определена'}")
    print(f"  Уверенность: {summary['confidence']:.2f}")
    print(f"  Стратегия: {summary['strategy']}")
    
    # Показываем ВСЕ сессии с оценками
    if all_scores:
        # Сортируем по убыванию релевантности
        sorted_sessions = sorted(all_scores.items(), key=lambda x: x[1]['relevance_score'], reverse=True)
        
        print("  Все сессии с оценками:")
        for session_id, info in sorted_sessions:
            relevance = info['relevance_score']
            status = info['status']
            keywords = info['matched_keywords']
            print(f"    Сессия {session_id}: {relevance:.3f} ({status}, ключевые слова: {keywords})")
    else:
        print("  Нет данных о релевантности сессий")


def main():
    """Основная функция"""
    # Пути к файлам
    data_file = "data/format_example.jsonl"
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
            prompt, session_analysis = process_dialog(rag, dialog)
            
            # Отладочная информация
            print(f"  Длина промпта: {len(prompt)} символов")
            if len(prompt) < 100:
                print(f"  ВНИМАНИЕ: Промпт слишком короткий!")
                print(f"  Полный промпт: {prompt}")
            
            # Сохраняем промпт
            save_prompt(dialog.id, dialog.question, prompt, output_dir)
            
            # Сохраняем анализ сессий
            save_session_analysis(dialog.id, dialog.question, session_analysis, output_dir)
            
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
    print("Созданы файлы:")
    print("  - prompt_N.txt - промпты для каждого диалога")
    print("  - session_analysis_N.json - анализ релевантности сессий")


if __name__ == "__main__":
    main()
