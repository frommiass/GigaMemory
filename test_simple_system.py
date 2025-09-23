#!/usr/bin/env python
"""
Простой тест системы GigaMemory без зависимостей от vllm
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from models import Message

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))


class MockModelInference:
    """Mock модель для тестирования"""
    
    def inference(self, messages: List[Message]) -> str:
        """Простая генерация ответа"""
        if not messages:
            return "Нет сообщений для обработки."
        
        # Анализируем последнее сообщение
        last_message = messages[-1].content.lower()
        
        if "зовут" in last_message:
            return "Вас зовут Александр Петров."
        elif "лет" in last_message:
            return "Вам 35 лет."
        elif "живете" in last_message or "живу" in last_message:
            return "Вы живете в Москве."
        elif "работаете" in last_message or "работаю" in last_message:
            return "Вы работаете программистом в Яндексе."
        else:
            return "Я понял ваше сообщение."


class SimpleMemorySystem:
    """Простая система памяти для тестирования"""
    
    def __init__(self):
        self.memory = {}
        self.filter_cache = {}
        self.stats = {
            'total_messages': 0,
            'filtered_messages': 0,
            'copypaste_filtered': 0,
            'sessions_created': 0,
            'facts_extracted': 0,
            'compression_ratio': 1.0
        }
        self.model = MockModelInference()
    
    def is_copy_paste_content(self, text: str) -> bool:
        """Простая проверка на копипаст"""
        copy_paste_indicators = [
            "переведи", "исправь", "напиши", "создай", "сгенерируй",
            "def ", "class ", "import ", "function", "код:", "код ",
            "статья", "текст:", "документ", "файл:", "ссылка:"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in copy_paste_indicators)
    
    def write_to_memory(self, messages: List[Message], dialogue_id: str):
        """Записывает сообщения в память"""
        if dialogue_id not in self.memory:
            self.memory[dialogue_id] = []
        
        filtered_messages = []
        for msg in messages:
            self.stats['total_messages'] += 1
            
            # Фильтруем копипаст
            if self.is_copy_paste_content(msg.content):
                self.stats['copypaste_filtered'] += 1
                continue
            
            filtered_messages.append(msg)
            self.stats['filtered_messages'] += 1
        
        self.memory[dialogue_id].extend(filtered_messages)
        
        # Подсчитываем сессии
        session_ids = set(msg.session_id for msg in filtered_messages)
        self.stats['sessions_created'] = len(session_ids)
        
        # Имитируем извлечение фактов
        self.stats['facts_extracted'] += len(filtered_messages) // 3
    
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос используя память"""
        if dialogue_id not in self.memory:
            return "Нет информации в памяти."
        
        # Используем все сообщения из памяти как контекст
        context_messages = self.memory[dialogue_id]
        
        # Создаем сообщение с вопросом
        question_msg = Message(
            role="user",
            content=question,
            session_id="question"
        )
        
        # Добавляем контекст
        all_messages = context_messages + [question_msg]
        
        return self.model.inference(all_messages)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        return self.stats.copy()
    
    def clear_memory(self, dialogue_id: str):
        """Очищает память диалога"""
        if dialogue_id in self.memory:
            del self.memory[dialogue_id]


def load_dialogue(file_path: str) -> Dict[str, Any]:
    """Загружает диалог из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """Обрабатывает диалог с помощью простой системы памяти"""
    try:
        # Создаем простую систему памяти
        system = SimpleMemorySystem()
        
        # Обрабатываем сессии
        dialogue_id = dialogue["id"]
        total_messages = 0
        
        for session in dialogue["sessions"]:
            messages = []
            for msg_data in session["messages"]:
                message = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    session_id=session["id"]
                )
                messages.append(message)
                total_messages += 1
            
            # Записываем в память
            system.write_to_memory(messages, dialogue_id)
        
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
            "total_messages": total_messages
        }
        
    except Exception as e:
        print(f"Ошибка при обработке диалога: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dialogue_id": dialogue["id"],
            "error": str(e)
        }


def main():
    """Основная функция тестирования"""
    print("🚀 Запуск простого теста системы GigaMemory")
    
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
    print(f"  Обработано сообщений: {result['stats']['filtered_messages']}")
    print(f"  Отфильтровано копипаста: {result['stats']['copypaste_filtered']}")
    print(f"  Создано сессий: {result['stats']['sessions_created']}")
    print(f"  Извлечено фактов: {result['stats']['facts_extracted']}")
    print(f"  Коэффициент сжатия: {result['stats']['compression_ratio']:.2f}")
    
    # Сохраняем результаты
    output_file = Path("simple_test_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в {output_file}")
    print("✅ Простой тест завершен успешно!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
