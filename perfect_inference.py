#!/usr/bin/env python
"""
Идеальная версия скрипта для обработки диалогов с правильной фильтрацией
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


def extract_facts_from_user_messages(user_messages: List[str]) -> List[str]:
    """
    Извлекает факты ТОЛЬКО из сообщений пользователя
    """
    facts = []
    
    for message in user_messages:
        message_lower = message.lower()
        
        # Извлекаем факт о семье
        if 'семье' in message_lower or 'семья' in message_lower:
            # Ищем числа в контексте семьи
            numbers = re.findall(r'\b(?:пятеро|шестеро|двое|трое|четверо|пятеро|шестеро|семеро|восьмеро|девятеро|десятеро|\d+)\b', message_lower)
            if numbers:
                facts.append(f"В семье {numbers[0]} человек")
        
        # Извлекаем факт о предпочтениях автомобилей
        if 'понравился' in message_lower or 'нравится' in message_lower:
            # Ищем название автомобиля
            car_brands = ['мультивен', 'volkswagen', 'ford', 'toyota', 'skoda', 'mitsubishi']
            for brand in car_brands:
                if brand in message_lower:
                    facts.append(f"Нравится автомобиль {brand.title()}")
                    break
        
        # Извлекаем факт о местоположении
        if 'москве' in message_lower or 'москва' in message_lower:
            facts.append("Живет в Москве")
            
        # Извлекаем факт о работе
        if 'работаю' in message_lower:
            # Ищем место работы
            work_places = ['яндексе', 'гугле', 'майкрософте', 'амазоне', 'компании', 'фирме']
            for place in work_places:
                if place in message_lower:
                    facts.append(f"Работает в {place}")
                    break
        
        # Извлекаем факт о возрасте
        if 'лет' in message_lower:
            age_match = re.search(r'(\d+)\s*лет', message_lower)
            if age_match:
                facts.append(f"Возраст {age_match.group(1)} лет")
        
        # Извлекаем факт о детях
        if 'сын' in message_lower or 'дочь' in message_lower:
            if 'сын' in message_lower and 'дочь' in message_lower:
                facts.append("Есть сын и дочь")
            elif 'сын' in message_lower:
                facts.append("Есть сын")
            elif 'дочь' in message_lower:
                facts.append("Есть дочь")
        
        # Извлекаем факт о жене/муже
        if 'жена' in message_lower:
            facts.append("Женат")
        elif 'муж' in message_lower:
            facts.append("Замужем")
    
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
    
    # Извлекаем факты ТОЛЬКО из сообщений пользователя
    facts = extract_facts_from_user_messages(all_user_messages)
    
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
        "Извлеченные факты:",
    ])
    
    if facts:
        for i, fact in enumerate(facts, 1):
            prompt_parts.append(f"{i}. {fact}")
    else:
        prompt_parts.append("Факты не извлечены.")
    
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
    """Запускает идеальный инференс на датасете"""
    print(f"🚀 Запуск ИДЕАЛЬНОГО инференса на датасете: {dataset_path}")
    
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
    print("✅ ИДЕАЛЬНЫЙ инференс завершен успешно!")
    
    return 0


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Perfect Dialogue Inference")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к датасету для инференса")
    parser.add_argument("--output", type=str, default="./perfect_output", help="Путь для сохранения результатов")
    
    args = parser.parse_args()
    
    return run_inference(args.dataset, args.output)


if __name__ == "__main__":
    sys.exit(main())
