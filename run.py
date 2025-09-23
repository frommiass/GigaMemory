#!/usr/bin/env python
"""
Основной скрипт для запуска системы GigaMemory
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

def load_dialogue(file_path: str) -> Dict[str, Any]:
    """Загружает диалог из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_dialogue(dialogue: Dict[str, Any], model_path: str = None) -> Dict[str, Any]:
    """Обрабатывает диалог с помощью системы памяти"""
    try:
        from submit.model_inference import SubmitModelWithMemory
        
        # Создаем систему с оптимизированной памятью
        system = SubmitModelWithMemory(model_path or "mock_model")
        
        # Обрабатываем сессии
        dialogue_id = dialogue["id"]
        total_messages = 0
        filtered_messages = 0
        
        for session in dialogue["sessions"]:
            messages = []
            for msg_data in session["messages"]:
                from models import Message
                message = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    session_id=session["id"]
                )
                messages.append(message)
                total_messages += 1
            
            # Записываем в память
            system.write_to_memory(messages, dialogue_id)
            filtered_messages += len(messages)
        
        # Получаем статистику
        stats = system.get_stats()
        
        # Генерируем ответ на вопрос
        question = dialogue.get("question", "Как меня зовут?")
        answer = system.answer_to_question(dialogue_id, question)
        
        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "answer": answer,
            "stats": stats,
            "total_messages": total_messages,
            "filtered_messages": filtered_messages
        }
        
    except Exception as e:
        print(f"Ошибка при обработке диалога: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dialogue_id": dialogue["id"],
            "error": str(e)
        }


def run_test():
    """Запускает тест с большим диалогом"""
    print("🚀 Запуск теста системы GigaMemory с большим диалогом")
    
    # Проверяем наличие тестового диалога
    test_file = Path("test_dialogue_100k.jsonl")
    if not test_file.exists():
        print("❌ Тестовый диалог не найден. Запустите сначала: python validate_system.py")
        return 1
    
    # Загружаем диалог
    print("📖 Загружаем тестовый диалог...")
    dialogue = load_dialogue(str(test_file))
    print(f"✅ Загружен диалог: {dialogue['id']} с {len(dialogue['sessions'])} сессиями")
    
    # Обрабатываем диалог
    print("⚙️ Обрабатываем диалог...")
    result = process_dialogue(dialogue)
    
    if "error" in result:
        print(f"❌ Ошибка: {result['error']}")
        return 1
    
    # Выводим результаты
    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"Диалог ID: {result['dialogue_id']}")
    print(f"Вопрос: {result['question']}")
    print(f"Ответ: {result['answer']}")
    print(f"\nСтатистика обработки:")
    print(f"  Всего сообщений: {result['total_messages']}")
    print(f"  Обработано сообщений: {result['filtered_messages']}")
    print(f"  Отфильтровано копипаста: {result['stats'].get('copypaste_filtered', 0)}")
    print(f"  Создано сессий: {result['stats'].get('sessions_created', 0)}")
    print(f"  Извлечено фактов: {result['stats'].get('facts_extracted', 0)}")
    print(f"  Коэффициент сжатия: {result['stats'].get('compression_ratio', 1.0):.2f}")
    
    # Сохраняем результаты
    output_file = Path("test_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в {output_file}")
    print("✅ Тест завершен успешно!")
    
    return 0


def run_inference(dataset_path: str, output_path: str, model_path: str):
    """Запускает инференс на датасете"""
    print(f"🚀 Запуск инференса на датасете: {dataset_path}")
    
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
    for i, dialogue in enumerate(dialogues):
        print(f"⚙️ Обрабатываем диалог {i+1}/{len(dialogues)}: {dialogue.get('id', 'unknown')}")
        result = process_dialogue(dialogue, model_path)
        results.append(result)
    
    # Сохраняем результаты
    output_file = output_dir / "results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"💾 Результаты сохранены в {output_file}")
    print("✅ Инференс завершен успешно!")
    
    return 0


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="GigaMemory System Runner")
    parser.add_argument("--test", action="store_true", help="Запустить тест с большим диалогом")
    parser.add_argument("--dataset", type=str, help="Путь к датасету для инференса")
    parser.add_argument("--output", type=str, default="./output", help="Путь для сохранения результатов")
    parser.add_argument("--model", type=str, default="/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16", help="Путь к модели")
    
    args = parser.parse_args()
    
    if args.test:
        return run_test()
    elif args.dataset:
        return run_inference(args.dataset, args.output, args.model)
    else:
        print("❌ Укажите --test для тестирования или --dataset для инференса")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
