#!/usr/bin/env python3
"""
Скрипт для извлечения только слов пользователя из диалогов
Сохраняет слова в отдельные файлы, пронумерованные по сессиям
"""
import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime


def load_dialogue(file_path: str) -> Dict[str, Any]:
    """Загружает диалог из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dialogues_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Загружает диалоги из JSONL файла"""
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dialogues.append(json.loads(line))
    return dialogues


def extract_words_from_text(text: str) -> List[str]:
    """
    Извлекает слова из текста, очищая от знаков препинания
    """
    if not text or not isinstance(text, str):
        return []
    
    # Убираем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Извлекаем слова (только буквы и цифры)
    words = re.findall(r'\b[а-яёa-z0-9]+\b', text.lower())
    
    return words


def is_user_message(message: Dict[str, Any]) -> bool:
    """Проверяет, является ли сообщение сообщением пользователя"""
    role = message.get('role', '').lower()
    return role in ['user', 'пользователь', 'human']


def is_copy_paste_content(content: str) -> bool:
    """
    Проверяет, является ли контент копипастом (длинные тексты, вопросы, рекомендации)
    """
    if not content or len(content) < 10:
        return False
    
    content_lower = content.lower()
    
    # Маркеры рекомендаций и вопросов
    recommendation_markers = [
        'посоветуй', 'расскажи', 'что можно', 'куда поехать',
        'чем заняться', 'побольше о', 'с каким', 'где можно',
        'а он дорогой', 'лучше взять', 'не хочу на авито',
        'какой', 'какая', 'какие', 'как', 'что', 'где', 'когда',
        'почему', 'куда', 'чем', 'расскажи', 'объясни',
        'откуда', 'откуда этот', 'расскажи принцип', 'во всех подробностях'
    ]
    
    # Проверяем на маркеры рекомендаций
    for marker in recommendation_markers:
        if marker in content_lower:
            return True
    
    # Длинные сообщения (>200 символов) считаем копипастом
    if len(content) > 200:
        return True
    
    # Проверяем на вопросы
    question_words = ['?', 'что ', 'где ', 'как ', 'когда ', 'почему ', 
                     'куда ', 'чем ', 'какой ', 'какая ', 'какие ']
    is_question = any(q in content_lower for q in question_words)
    
    # Если это вопрос, считаем копипастом
    if is_question:
        return True
    
    return False


def process_dialogue(dialogue: Dict[str, Any], min_words: int = 3) -> Dict[str, List[str]]:
    """
    Обрабатывает диалог и извлекает слова пользователя по сессиям
    
    Args:
        dialogue: Диалог для обработки
        min_words: Минимальное количество слов в сообщении для включения
        
    Returns:
        Словарь {session_id: [список слов]}
    """
    session_words = {}
    
    dialogue_id = dialogue.get('id', 'unknown')
    sessions = dialogue.get('sessions', [])
    
    print(f"Обрабатываем диалог {dialogue_id} с {len(sessions)} сессиями")
    
    for session in sessions:
        session_id = session.get('id', 'unknown')
        messages = session.get('messages', [])
        
        session_word_list = []
        
        for message in messages:
            # Проверяем, что это сообщение пользователя
            if not is_user_message(message):
                continue
            
            content = message.get('content', '')
            if not content:
                continue
            
            # Пропускаем копипаст
            if is_copy_paste_content(content):
                continue
            
            # Извлекаем слова
            words = extract_words_from_text(content)
            
            # Добавляем только если достаточно слов
            if len(words) >= min_words:
                session_word_list.extend(words)
        
        # Сохраняем только если есть слова
        if session_word_list:
            session_words[session_id] = session_word_list
            print(f"  Сессия {session_id}: {len(session_word_list)} слов")
    
    return session_words


def save_session_words(session_words: Dict[str, List[str]], 
                      dialogue_id: str, 
                      output_dir: Path,
                      format_type: str = 'txt'):
    """
    Сохраняет слова сессий в отдельные файлы
    
    Args:
        session_words: Словарь {session_id: [список слов]}
        dialogue_id: ID диалога
        output_dir: Директория для сохранения
        format_type: Формат файла ('txt' или 'json')
    """
    dialogue_dir = output_dir / f"dialogue_{dialogue_id}"
    dialogue_dir.mkdir(exist_ok=True, parents=True)
    
    for session_id, words in session_words.items():
        if format_type == 'txt':
            # Сохраняем как текстовый файл
            filename = f"session_{session_id}.txt"
            filepath = dialogue_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Сессия {session_id} - Диалог {dialogue_id}\n")
                f.write(f"Всего слов: {len(words)}\n")
                f.write("=" * 50 + "\n\n")
                
                # Записываем слова по 10 в строку
                for i in range(0, len(words), 10):
                    line_words = words[i:i+10]
                    f.write(" ".join(line_words) + "\n")
        
        elif format_type == 'json':
            # Сохраняем как JSON файл
            filename = f"session_{session_id}.json"
            filepath = dialogue_dir / filename
            
            session_data = {
                "dialogue_id": dialogue_id,
                "session_id": session_id,
                "total_words": len(words),
                "words": words,
                "unique_words": list(set(words)),
                "unique_count": len(set(words))
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранено {len(session_words)} сессий в {dialogue_dir}")


def create_summary_file(all_session_words: Dict[str, Dict[str, List[str]]], 
                       output_dir: Path):
    """
    Создает сводный файл со статистикой
    """
    summary_file = output_dir / "summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("СВОДКА ПО ИЗВЛЕЧЕННЫМ СЛОВАМ ПОЛЬЗОВАТЕЛЯ\n")
        f.write("=" * 50 + "\n\n")
        
        total_dialogues = len(all_session_words)
        total_sessions = sum(len(sessions) for sessions in all_session_words.values())
        total_words = sum(
            len(words) 
            for sessions in all_session_words.values() 
            for words in sessions.values()
        )
        
        f.write(f"Всего диалогов: {total_dialogues}\n")
        f.write(f"Всего сессий: {total_sessions}\n")
        f.write(f"Всего слов: {total_words}\n\n")
        
        # Статистика по диалогам
        f.write("СТАТИСТИКА ПО ДИАЛОГАМ:\n")
        f.write("-" * 30 + "\n")
        
        for dialogue_id, sessions in all_session_words.items():
            dialogue_words = sum(len(words) for words in sessions.values())
            f.write(f"Диалог {dialogue_id}: {len(sessions)} сессий, {dialogue_words} слов\n")
            
            for session_id, words in sessions.items():
                unique_words = len(set(words))
                f.write(f"  Сессия {session_id}: {len(words)} слов ({unique_words} уникальных)\n")
        
        # Топ слов
        f.write("\nТОП-20 САМЫХ ЧАСТЫХ СЛОВ:\n")
        f.write("-" * 30 + "\n")
        
        word_counts = {}
        for sessions in all_session_words.values():
            for words in sessions.values():
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for word, count in top_words:
            f.write(f"{word}: {count}\n")
    
    print(f"Сводка сохранена в {summary_file}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Извлечение слов пользователя из диалогов")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному файлу (JSON или JSONL)")
    parser.add_argument("--output", type=str, default="./user_words_output", help="Путь для сохранения результатов")
    parser.add_argument("--format", type=str, choices=['txt', 'json'], default='txt', help="Формат выходных файлов")
    parser.add_argument("--min-words", type=int, default=3, help="Минимальное количество слов в сообщении")
    
    args = parser.parse_args()
    
    # Создаем выходную директорию
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Извлечение слов пользователя из файла: {args.input}")
    print(f"📁 Выходная директория: {args.output}")
    print(f"📄 Формат файлов: {args.format}")
    print(f"🔢 Минимум слов в сообщении: {args.min_words}")
    print()
    
    # Определяем тип файла и загружаем данные
    input_path = Path(args.input)
    if input_path.suffix.lower() == '.jsonl':
        dialogues = load_dialogues_from_jsonl(str(input_path))
    else:
        # Предполагаем JSON файл
        dialogues = [load_dialogue(str(input_path))]
    
    print(f"📖 Загружено {len(dialogues)} диалогов")
    
    # Обрабатываем каждый диалог
    all_session_words = {}
    
    for dialogue in dialogues:
        dialogue_id = dialogue.get('id', 'unknown')
        session_words = process_dialogue(dialogue, args.min_words)
        
        if session_words:
            all_session_words[dialogue_id] = session_words
            save_session_words(session_words, dialogue_id, output_dir, args.format)
    
    # Создаем сводный файл
    if all_session_words:
        create_summary_file(all_session_words, output_dir)
    
    print(f"\n✅ Извлечение завершено!")
    print(f"📊 Обработано диалогов: {len(all_session_words)}")
    print(f"📁 Результаты сохранены в: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
