#!/usr/bin/env python
"""
Упрощенный скрипт для обработки диалогов без vllm
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

def load_dialogue(file_path: str) -> Dict[str, Any]:
    """Загружает диалог из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_facts_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Извлекает факты из сообщений"""
    facts = []
    
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        
        # Простое извлечение фактов - ищем утверждения пользователя
        if role == "user" and content:
            # Разбиваем на предложения и ищем факты
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Минимальная длина факта
                    facts.append(sentence)
    
    return facts


def create_prompt_from_dialogue(dialogue: Dict[str, Any]) -> str:
    """Создает промпт на основе диалога"""
    dialogue_id = dialogue.get("id", "unknown")
    question = dialogue.get("question", "Как меня зовут?")
    
    # Собираем все сообщения
    all_messages = []
    for session in dialogue.get("sessions", []):
        for msg in session.get("messages", []):
            all_messages.append(msg)
    
    # Извлекаем факты
    facts = extract_facts_from_messages(all_messages)
    
    # Создаем промпт
    prompt_parts = [
        f"Диалог ID: {dialogue_id}",
        f"Вопрос: {question}",
        "",
        "Контекст диалога:",
    ]
    
    # Добавляем последние сообщения как контекст
    recent_messages = all_messages[-10:] if len(all_messages) > 10 else all_messages
    for msg in recent_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    
    prompt_parts.extend([
        "",
        "Извлеченные факты:",
    ])
    
    for i, fact in enumerate(facts[:5], 1):  # Берем только первые 5 фактов
        prompt_parts.append(f"{i}. {fact}")
    
    prompt_parts.extend([
        "",
        f"На основе предоставленного контекста ответь на вопрос: {question}"
    ])
    
    return "\n".join(prompt_parts)


def process_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """Обрабатывает диалог и создает промпт"""
    try:
        dialogue_id = dialogue.get("id", "unknown")
        question = dialogue.get("question", "Как меня зовут?")
        
        # Создаем промпт
        prompt = create_prompt_from_dialogue(dialogue)
        
        # Подсчитываем статистику
        total_messages = 0
        for session in dialogue.get("sessions", []):
            total_messages += len(session.get("messages", []))
        
        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "prompt": prompt,
            "total_messages": total_messages,
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
    """Запускает инференс на датасете"""
    print(f"🚀 Запуск упрощенного инференса на датасете: {dataset_path}")
    
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
    print("✅ Инференс завершен успешно!")
    
    return 0


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Simple Dialogue Inference")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к датасету для инференса")
    parser.add_argument("--output", type=str, default="./simple_output", help="Путь для сохранения результатов")
    
    args = parser.parse_args()
    
    return run_inference(args.dataset, args.output)


if __name__ == "__main__":
    sys.exit(main())


