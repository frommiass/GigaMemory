#!/usr/bin/env python
"""
Простой скрипт для показа извлеченных фактов из диалогов
"""
import json
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

def load_leaderboard_data(file_path: str):
    """Загружает данные лидерборда из JSONL файла"""
    dialogues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    dialogue = json.loads(line)
                    dialogues.append(dialogue)
                except json.JSONDecodeError as e:
                    print(f"❌ Ошибка парсинга строки {line_num}: {e}")
    
    return dialogues

def extract_facts_from_dialogue(dialogue):
    """Извлекает факты из диалога"""
    facts = []
    
    for session in dialogue.get('sessions', []):
        session_id = session.get('id', 'unknown')
        
        for msg in session.get('messages', []):
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                
                # Факты о собаке
                if any(word in content for word in ['собака', 'пес', 'пёс', 'собачку']):
                    fact_info = {
                        'type': 'dog',
                        'content': msg.get('content', ''),
                        'session_id': session_id,
                        'message_id': f"session_{session_id}_msg_{len(facts)}"
                    }
                    facts.append(fact_info)
                
                # Факты о спорте
                elif any(word in content for word in ['спорт', 'тренировка', 'бег', 'плавание', 'футбол', 'теннис']):
                    fact_info = {
                        'type': 'sport',
                        'content': msg.get('content', ''),
                        'session_id': session_id,
                        'message_id': f"session_{session_id}_msg_{len(facts)}"
                    }
                    facts.append(fact_info)
                
                # Факты о работе
                elif any(word in content for word in ['работа', 'профессия', 'должность', 'компания', 'офис']):
                    fact_info = {
                        'type': 'work',
                        'content': msg.get('content', ''),
                        'session_id': session_id,
                        'message_id': f"session_{session_id}_msg_{len(facts)}"
                    }
                    facts.append(fact_info)
                
                # Факты о курении
                elif any(word in content for word in ['сигареты', 'курю', 'курить', 'табак', 'никотин']):
                    fact_info = {
                        'type': 'smoking',
                        'content': msg.get('content', ''),
                        'session_id': session_id,
                        'message_id': f"session_{session_id}_msg_{len(facts)}"
                    }
                    facts.append(fact_info)
                
                # Общие личные факты
                elif any(word in content for word in ['зовут', 'живу', 'родился', 'женат', 'дети']):
                    fact_info = {
                        'type': 'personal',
                        'content': msg.get('content', ''),
                        'session_id': session_id,
                        'message_id': f"session_{session_id}_msg_{len(facts)}"
                    }
                    facts.append(fact_info)
    
    return facts

def main():
    """Основная функция"""
    print("🔍 ПОКАЗ ИЗВЛЕЧЕННЫХ ФАКТОВ ИЗ ДИАЛОГОВ")
    print("=" * 60)
    
    # Загружаем диалоги
    dialogues = load_leaderboard_data("data/format_example.jsonl")
    
    for i, dialogue in enumerate(dialogues, 1):
        dialogue_id = dialogue.get('id', f'dialogue_{i}')
        question = dialogue.get('question', 'Нет вопроса')
        
        print(f"\n📋 ДИАЛОГ {i} (ID: {dialogue_id})")
        print(f"❓ Вопрос: {question}")
        print("-" * 60)
        
        # Извлекаем факты
        facts = extract_facts_from_dialogue(dialogue)
        
        if not facts:
            print("❌ Факты не найдены")
            continue
        
        # Группируем факты по типам
        facts_by_type = {}
        for fact in facts:
            fact_type = fact['type']
            if fact_type not in facts_by_type:
                facts_by_type[fact_type] = []
            facts_by_type[fact_type].append(fact)
        
        # Показываем факты по типам
        for fact_type, type_facts in facts_by_type.items():
            print(f"\n📌 {fact_type.upper()} ({len(type_facts)} фактов):")
            
            for j, fact in enumerate(type_facts, 1):
                print(f"  {j}. Сессия {fact['session_id']}: {fact['content']}")
        
        print(f"\n📊 Всего извлечено фактов: {len(facts)}")
        print("=" * 60)

if __name__ == "__main__":
    main()
