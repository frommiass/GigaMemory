#!/usr/bin/env python3
"""
Простой скрипт для извлечения фактов из диалогов
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def extract_simple_facts(text: str) -> Dict[str, List[str]]:
    """Извлекает простые факты из текста"""
    facts = {
        'name': [],
        'age': [],
        'profession': [],
        'location': [],
        'family': [],
        'hobbies': [],
        'pets': []
    }
    
    text_lower = text.lower()
    
    # Имя - ищем простые паттерны
    name_patterns = [
        r'меня зовут ([а-яё]+)',
        r'мое имя ([а-яё]+)',
        r'зовут ([а-яё]+)',
        r'я ([а-яё]+)'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) >= 2 and len(match) <= 20:  # Разумная длина имени
                facts['name'].append(match.title())
    
    # Возраст - ищем числа
    age_patterns = [
        r'мне (\d+) лет',
        r'исполнилось (\d+)',
        r'возраст (\d+)',
        r'(\d+) лет'
    ]
    
    for pattern in age_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            age = int(match)
            if 1 <= age <= 120:  # Разумный возраст
                facts['age'].append(str(age))
    
    # Профессия - ищем простые слова
    profession_keywords = [
        'программист', 'врач', 'учитель', 'инженер', 'дизайнер', 'менеджер',
        'бухгалтер', 'юрист', 'повар', 'водитель', 'строитель', 'сварщик',
        'депутат', 'блогер', 'фотограф', 'музыкант', 'художник', 'писатель'
    ]
    
    for keyword in profession_keywords:
        if keyword in text_lower:
            facts['profession'].append(keyword.title())
    
    # Местоположение - ищем города и регионы
    location_keywords = [
        'москва', 'санкт-петербург', 'екатеринбург', 'новосибирск', 'казань',
        'нижний новгород', 'челябинск', 'самара', 'омск', 'ростов-на-дону',
        'сибирь', 'урал', 'дон', 'волга', 'байкал'
    ]
    
    for keyword in location_keywords:
        if keyword in text_lower:
            facts['location'].append(keyword.title())
    
    # Семья - ищем упоминания родственников
    family_keywords = [
        'жена', 'муж', 'дочь', 'сын', 'мама', 'папа', 'бабушка', 'дедушка',
        'сестра', 'брат', 'племянница', 'племянник'
    ]
    
    for keyword in family_keywords:
        if keyword in text_lower:
            facts['family'].append(keyword.title())
    
    # Увлечения - ищем хобби
    hobby_keywords = [
        'фотография', 'музыка', 'спорт', 'танцы', 'чтение', 'путешествия',
        'рыбалка', 'охота', 'садоводство', 'готовка', 'рисование', 'пение',
        'футбол', 'теннис', 'плавание', 'бег', 'йога', 'танго'
    ]
    
    for keyword in hobby_keywords:
        if keyword in text_lower:
            facts['hobbies'].append(keyword.title())
    
    # Питомцы - ищем животных
    pet_keywords = [
        'собака', 'кошка', 'кот', 'пес', 'котенок', 'щенок', 'хомяк',
        'птица', 'рыбка', 'черепаха', 'кролик'
    ]
    
    for keyword in pet_keywords:
        if keyword in text_lower:
            facts['pets'].append(keyword.title())
    
    # Убираем дубликаты
    for category in facts:
        facts[category] = list(set(facts[category]))
    
    return facts

def create_simple_prompt(dialog_id: int, facts: Dict[str, List[str]], user_messages: List[str]) -> str:
    """Создает простой промпт с фактами"""
    
    prompt = f"""# Факты о пользователе из диалога {dialog_id}

## Извлеченные факты:

"""
    
    # Добавляем только непустые категории
    category_names = {
        'name': 'Имя',
        'age': 'Возраст',
        'profession': 'Профессия', 
        'location': 'Местоположение',
        'family': 'Семья',
        'hobbies': 'Увлечения',
        'pets': 'Питомцы'
    }
    
    has_facts = False
    for category, values in facts.items():
        if values:
            has_facts = True
            category_name = category_names[category]
            prompt += f"**{category_name}:** {', '.join(values)}\n"
    
    if not has_facts:
        prompt += "Факты не найдены в диалоге.\n"
    
    prompt += f"""

## Примеры сообщений пользователя:
{chr(10).join(user_messages[:3])}

## Инструкции:
1. Используйте эти факты для персонализированных ответов
2. Помните контекст диалога
3. Обновляйте факты при получении новой информации

---
*Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return prompt

def main():
    """Основная функция"""
    print("🎯 Простое извлечение фактов из диалогов")
    print("="*60)
    
    # Путь к файлу данных
    data_file = "data/format_example.jsonl"
    
    if not Path(data_file).exists():
        print(f"❌ Файл {data_file} не найден!")
        return
    
    try:
        # Загружаем диалоги
        print("📂 Загружаем диалоги...")
        dialogs = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        dialogs.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Ошибка при загрузке строки {line_num}: {e}")
                        continue
        
        print(f"✅ Загружено {len(dialogs)} диалогов")
        
        # Создаем папку для результатов
        output_dir = Path("simple_facts")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🔧 Извлекаем факты...")
        
        # Обрабатываем каждый диалог
        for i, dialog in enumerate(dialogs, 1):
            print(f"  • Обрабатываем диалог {i}...")
            
            # Собираем все сообщения пользователя
            user_messages = []
            for session in dialog.get('sessions', []):
                for message in session.get('messages', []):
                    if message.get('role') == 'user':
                        content = message.get('content', '')
                        if content:
                            user_messages.append(content)
            
            # Объединяем все сообщения
            all_text = ' '.join(user_messages)
            
            # Извлекаем факты
            facts = extract_simple_facts(all_text)
            
            # Создаем промпт
            prompt = create_simple_prompt(i, facts, user_messages)
            
            # Сохраняем промпт
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dialog_{i}_facts_{timestamp}.txt"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            print(f"    ✅ Сохранен: {filename}")
        
        print(f"\n🎉 Готово! Создано {len(dialogs)} файлов с фактами")
        print(f"📁 Папка: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
