#!/usr/bin/env python
"""
Правильная версия скрипта с извлечением фактов по теме вопроса
"""
import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

def load_dialogue(file_path: str) -> Dict[str, Any]:
    """Загружает диалог из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_copy_paste_content(content: str) -> bool:
    """
    Улучшенная проверка на копипаст
    """
    content_lower = content.lower()
    
    # КРИТИЧНО: Запросы на рекомендации это НЕ личная информация
    recommendation_markers = [
        'посоветуй', 'расскажи', 'что можно', 'куда поехать',
        'чем заняться', 'побольше о', 'с каким', 'где можно',
        'а он дорогой', 'лучше взять', 'не хочу на авито',
        'какой', 'какая', 'какие', 'как', 'что', 'где', 'когда',
        'почему', 'куда', 'чем', 'расскажи', 'объясни',
        'откуда', 'откуда этот', 'расскажи принцип', 'во всех подробностях',
        'мне интересно', 'хочу узнать', 'расскажи мне'
    ]
    
    for marker in recommendation_markers:
        if marker in content_lower:
            return True
    
    # Длинные сообщения (>150 символов) почти всегда копипаст
    if len(content) > 150:
        return True
        
    # Проверяем на вопросы
    question_words = ['?', 'что ', 'где ', 'как ', 'когда ', 'почему ', 
                     'куда ', 'чем ', 'какой ', 'какая ', 'какие ']
    is_question = any(q in content_lower for q in question_words)
    
    if is_question:
        return True
        
    return False


def contains_personal_info(content: str) -> bool:
    """
    Проверяет содержит ли сообщение личную информацию
    """
    content_lower = content.lower()
    
    # Индикаторы личной информации
    personal_indicators = [
        'я', 'меня', 'мой', 'моя', 'мне', 'у меня',
        'мы', 'нас', 'наш', 'наша', 'нам', 'у нас',
        'семья', 'семье', 'детей', 'жена', 'муж',
        'сын', 'дочь', 'ребенок', 'дети',
        'работаю', 'живу', 'езжу', 'имею', 'владею'
    ]
    
    return any(ind in content_lower for ind in personal_indicators)


def extract_user_messages_only(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Извлекает ТОЛЬКО сообщения пользователя с фильтрацией
    """
    user_messages = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        
        # КРИТИЧНО: Только сообщения USER!
        if role != "user":
            continue
            
        # Пропускаем пустые сообщения
        if not content:
            continue
            
        # Фильтруем копипаст
        if is_copy_paste_content(content):
            continue
            
        # Проверяем длину
        if len(content) < 15 or len(content) > 200:
            continue
            
        # Проверяем что это личная информация
        if contains_personal_info(content):
            user_messages.append(content)
    
    return user_messages


def extract_facts_by_question_topic(user_messages: List[str], question: str) -> List[str]:
    """
    Извлекает факты ТОЛЬКО по теме вопроса
    """
    facts = []
    question_lower = question.lower()
    
    # Определяем тему вопроса
    if 'спорт' in question_lower or 'занимаюсь' in question_lower:
        # Ищем информацию о спорте
        for message in user_messages:
            message_lower = message.lower()
            if 'спорт' in message_lower:
                if 'футбол' in message_lower:
                    facts.append("Занимается футболом")
                elif 'плавание' in message_lower:
                    facts.append("Занимается плаванием")
                elif 'бег' in message_lower:
                    facts.append("Занимается бегом")
                elif 'костюм' in message_lower and 'отказываюсь' in message_lower:
                    facts.append("Не носит спортивные костюмы")
                    
    elif 'работа' in question_lower or 'работаю' in question_lower:
        # Ищем информацию о работе
        for message in user_messages:
            message_lower = message.lower()
            if 'работаю' in message_lower:
                if 'яндексе' in message_lower:
                    facts.append("Работает в Яндексе")
                elif 'программист' in message_lower:
                    facts.append("Работает программистом")
                elif 'стоматолог' in message_lower:
                    facts.append("Работает стоматологом")
                    
    elif 'собака' in question_lower or 'порода' in question_lower:
        # Ищем информацию о собаке
        for message in user_messages:
            message_lower = message.lower()
            if 'собака' in message_lower or 'пес' in message_lower or 'пёс' in message_lower:
                if 'лабрадор' in message_lower:
                    facts.append("Собака породы лабрадор")
                elif 'овчарка' in message_lower:
                    facts.append("Собака породы овчарка")
                elif 'хаски' in message_lower:
                    facts.append("Собака породы хаски")
                elif 'мопс' in message_lower:
                    facts.append("Собака породы мопс")
                elif 'такса' in message_lower:
                    facts.append("Собака породы такса")
                    
    elif 'кошка' in question_lower or 'кот' in question_lower:
        # Ищем информацию о кошке
        for message in user_messages:
            message_lower = message.lower()
            if 'кошка' in message_lower or 'кот' in message_lower:
                if 'перс' in message_lower:
                    facts.append("Кошка персидской породы")
                elif 'британ' in message_lower:
                    facts.append("Кошка британской породы")
                elif 'сиам' in message_lower:
                    facts.append("Кошка сиамской породы")
                    
    elif 'машина' in question_lower or 'автомобиль' in question_lower:
        # Ищем информацию об автомобиле
        for message in user_messages:
            message_lower = message.lower()
            if 'машина' in message_lower or 'автомобиль' in message_lower:
                if 'тойота' in message_lower:
                    facts.append("Ездит на Toyota")
                elif 'мерседес' in message_lower:
                    facts.append("Ездит на Mercedes")
                elif 'бмв' in message_lower:
                    facts.append("Ездит на BMW")
                elif 'ауди' in message_lower:
                    facts.append("Ездит на Audi")
                    
    elif 'возраст' in question_lower or 'лет' in question_lower:
        # Ищем информацию о возрасте
        for message in user_messages:
            message_lower = message.lower()
            if 'лет' in message_lower:
                age_match = re.search(r'(\d+)\s*лет', message_lower)
                if age_match:
                    facts.append(f"Возраст {age_match.group(1)} лет")
                    
    elif 'имя' in question_lower or 'зовут' in question_lower:
        # Ищем информацию об имени
        for message in user_messages:
            message_lower = message.lower()
            if 'зовут' in message_lower or 'имя' in message_lower:
                # Ищем имя после "зовут" или "имя"
                name_match = re.search(r'(?:зовут|имя)\s+([а-яё]+)', message_lower)
                if name_match:
                    facts.append(f"Имя: {name_match.group(1)}")
    
    return facts


def create_prompt_from_dialogue(dialogue: Dict[str, Any]) -> str:
    """
    Создает промпт на основе диалога с правильной фильтрацией
    """
    dialogue_id = dialogue.get("id", "unknown")
    question = dialogue.get("question", "Как меня зовут?")
    
    # Собираем ТОЛЬКО сообщения пользователя из всех сессий
    all_user_messages = []
    for session in dialogue.get("sessions", []):
        session_messages = session.get("messages", [])
        user_messages = extract_user_messages_only(session_messages)
        all_user_messages.extend(user_messages)
    
    # Извлекаем факты ТОЛЬКО по теме вопроса
    facts = extract_facts_by_question_topic(all_user_messages, question)
    
    # Создаем промпт
    prompt_parts = [
        f"Диалог ID: {dialogue_id}",
        f"Вопрос: {question}",
        "",
        "Информация о пользователе:",
    ]
    
    # Добавляем только личные сообщения пользователя
    if all_user_messages:
        for i, msg in enumerate(all_user_messages[-5:], 1):  # Последние 5 сообщений
            prompt_parts.append(f"{i}. {msg}")
    else:
        prompt_parts.append("Личная информация не найдена в диалоге.")
    
    prompt_parts.extend([
        "",
        "Извлеченные факты по теме вопроса:",
    ])
    
    if facts:
        for i, fact in enumerate(facts, 1):
            prompt_parts.append(f"{i}. {fact}")
    else:
        prompt_parts.append("Факты по теме вопроса не найдены.")
    
    prompt_parts.extend([
        "",
        f"На основе предоставленной информации ответь на вопрос: {question}"
    ])
    
    return "\n".join(prompt_parts)


def process_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """Обрабатывает диалог и создает промпт с правильной фильтрацией"""
    try:
        dialogue_id = dialogue.get("id", "unknown")
        question = dialogue.get("question", "Как меня зовут?")
        
        # Создаем промпт с правильной фильтрацией
        prompt = create_prompt_from_dialogue(dialogue)
        
        # Подсчитываем статистику
        total_messages = 0
        user_messages_count = 0
        filtered_messages_count = 0
        
        for session in dialogue.get("sessions", []):
            session_messages = session.get("messages", [])
            total_messages += len(session_messages)
            
            # Подсчитываем сообщения пользователя
            for msg in session_messages:
                if msg.get("role") == "user":
                    user_messages_count += 1
                    content = msg.get("content", "").strip()
                    
                    # Подсчитываем отфильтрованные
                    if (len(content) >= 15 and len(content) <= 200 and 
                        not is_copy_paste_content(content) and 
                        contains_personal_info(content)):
                        filtered_messages_count += 1
        
        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "prompt": prompt,
            "total_messages": total_messages,
            "user_messages": user_messages_count,
            "filtered_messages": filtered_messages_count,
            "sessions_count": len(dialogue.get("sessions", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Ошибка при обработке диалога: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dialogue_id": dialogue.get("id", "unknown"),
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def run_inference(dataset_path: str, output_path: str):
    """Запускает правильный инференс на датасете"""
    print(f"🚀 Запуск ПРАВИЛЬНОГО инференса на датасете: {dataset_path}")
    
    # Создаем выходную директорию
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Загружаем датасет
    dialogues = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dialogues.append(json.loads(line))
    
    print(f"📖 Загружено {len(dialogues)} диалогов")
    
    # Обрабатываем каждый диалог
    results = []
    prompts = []
    
    for i, dialogue in enumerate(dialogues):
        print(f"⚙️ Обрабатываем диалог {i+1}/{len(dialogues)}: {dialogue.get('id', 'unknown')}")
        result = process_dialogue(dialogue)
        results.append(result)
        
        # Сохраняем промпт отдельно
        if "prompt" in result:
            prompts.append({
                "dialogue_id": result["dialogue_id"],
                "prompt": result["prompt"]
            })
    
    # Сохраняем результаты
    output_file = output_dir / "results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Сохраняем промпты отдельно
    prompts_file = output_dir / "prompts.jsonl"
    with open(prompts_file, 'w', encoding='utf-8') as f:
        for prompt_data in prompts:
            f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
    
    # Сохраняем промпты в отдельные файлы
    prompts_dir = output_dir / "prompt_files"
    prompts_dir.mkdir(exist_ok=True)
    
    for i, prompt_data in enumerate(prompts):
        prompt_file = prompts_dir / f"prompt_{prompt_data['dialogue_id']}.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_data["prompt"])
    
    print(f"💾 Результаты сохранены в {output_file}")
    print(f"💾 Промпты сохранены в {prompts_file}")
    print(f"💾 Отдельные файлы промптов сохранены в {prompts_dir}")
    print("✅ ПРАВИЛЬНЫЙ инференс завершен успешно!")
    
    return 0


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Correct Dialogue Inference")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к датасету для инференса")
    parser.add_argument("--output", type=str, default="./correct_output", help="Путь для сохранения результатов")
    
    args = parser.parse_args()
    
    return run_inference(args.dataset, args.output)


if __name__ == "__main__":
    sys.exit(main())


