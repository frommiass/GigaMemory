#!/usr/bin/env python3
"""
Очень простой скрипт для извлечения чистых фактов из диалогов
"""
import json
import re
from pathlib import Path
from datetime import datetime

def extract_clean_facts(text: str) -> dict:
    """Извлекает только чистые факты из текста"""
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
    
    # Имя - только четкие упоминания
    if 'меня зовут' in text_lower:
        match = re.search(r'меня зовут ([а-яё]+)', text_lower)
        if match:
            name = match.group(1).title()
            if len(name) >= 2 and len(name) <= 15:
                facts['name'].append(name)
    
    # Возраст - только четкие упоминания
    if 'мне' in text_lower and 'лет' in text_lower:
        match = re.search(r'мне (\d+) лет', text_lower)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                facts['age'].append(str(age))
    
    # Профессия - только четкие упоминания
    profession_words = ['программист', 'врач', 'учитель', 'инженер', 'дизайнер', 'менеджер',
                       'бухгалтер', 'юрист', 'повар', 'водитель', 'строитель', 'сварщик',
                       'депутат', 'блогер', 'фотограф', 'музыкант', 'художник', 'писатель']
    
    for word in profession_words:
        if word in text_lower:
            facts['profession'].append(word.title())
    
    # Местоположение - только четкие упоминания
    location_words = ['москва', 'санкт-петербург', 'екатеринбург', 'новосибирск', 'казань',
                     'нижний новгород', 'челябинск', 'самара', 'омск', 'ростов-на-дону',
                     'сибирь', 'урал', 'дон', 'волга', 'байкал']
    
    for word in location_words:
        if word in text_lower:
            facts['location'].append(word.title())
    
    # Семья - только четкие упоминания
    family_words = ['жена', 'муж', 'дочь', 'сын', 'мама', 'папа', 'бабушка', 'дедушка',
                   'сестра', 'брат', 'племянница', 'племянник']
    
    for word in family_words:
        if word in text_lower:
            facts['family'].append(word.title())
    
    # Увлечения - только четкие упоминания
    hobby_words = ['фотография', 'музыка', 'спорт', 'танцы', 'чтение', 'путешествия',
                  'рыбалка', 'охота', 'садоводство', 'готовка', 'рисование', 'пение',
                  'футбол', 'теннис', 'плавание', 'бег', 'йога', 'танго']
    
    for word in hobby_words:
        if word in text_lower:
            facts['hobbies'].append(word.title())
    
    # Питомцы - только четкие упоминания
    pet_words = ['собака', 'кошка', 'кот', 'пес', 'котенок', 'щенок', 'хомяк',
                'птица', 'рыбка', 'черепаха', 'кролик']
    
    for word in pet_words:
        if word in text_lower:
            facts['pets'].append(word.title())
    
    # Убираем дубликаты
    for category in facts:
        facts[category] = list(set(facts[category]))
    
    return facts

def create_clean_prompt(dialog_id: int, facts: dict, user_messages: list) -> str:
    """Создает чистый промпт с фактами"""
    
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
    print("🎯 Извлечение чистых фактов из диалогов")
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
        output_dir = Path("clean_facts")
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
            facts = extract_clean_facts(all_text)
            
            # Создаем промпт
            prompt = create_clean_prompt(i, facts, user_messages)
            
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
