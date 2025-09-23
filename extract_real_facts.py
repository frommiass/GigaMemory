#!/usr/bin/env python3
"""
Скрипт для извлечения реальных фактов из диалогов
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class RealFactExtractor:
    """Извлекает реальные факты из диалогов"""
    
    def __init__(self):
        # Паттерны для извлечения реальных фактов
        self.fact_patterns = {
            'name': [
                r'меня зовут ([А-Яа-я]+)',
                r'мое имя ([А-Яа-я]+)',
                r'зовут ([А-Яа-я]+)',
                r'я ([А-Яа-я]+)'
            ],
            'age': [
                r'мне (\d+) лет',
                r'исполнилось (\d+)',
                r'возраст (\d+)',
                r'(\d+) лет'
            ],
            'profession': [
                r'работаю ([А-Яа-я\s]+)',
                r'я ([А-Яа-я\s]+)',
                r'профессия ([А-Яа-я\s]+)',
                r'занимаюсь ([А-Яа-я\s]+)'
            ],
            'location': [
                r'живу в ([А-Яа-я\s]+)',
                r'находимся в ([А-Яа-я\s]+)',
                r'город ([А-Яа-я\s]+)',
                r'район ([А-Яа-я\s]+)'
            ],
            'family': [
                r'жена ([А-Яа-я]+)',
                r'муж ([А-Яа-я]+)',
                r'дочь ([А-Яа-я]+)',
                r'сын ([А-Яа-я]+)',
                r'дети ([А-Яа-я\s]+)'
            ],
            'hobbies': [
                r'увлекаюсь ([А-Яа-я\s]+)',
                r'хобби ([А-Яа-я\s]+)',
                r'занимаюсь ([А-Яа-я\s]+)',
                r'люблю ([А-Яа-я\s]+)'
            ],
            'pets': [
                r'собака ([А-Яа-я]+)',
                r'кошка ([А-Яа-я]+)',
                r'питомец ([А-Яа-я\s]+)'
            ]
        }
    
    def extract_facts_from_text(self, text: str) -> Dict[str, List[str]]:
        """Извлекает факты из текста"""
        facts = {category: [] for category in self.fact_patterns.keys()}
        text_lower = text.lower()
        
        for category, patterns in self.fact_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                # Фильтруем слишком короткие или длинные результаты
                filtered_matches = [match.strip() for match in matches 
                                   if 2 <= len(match.strip()) <= 50]
                facts[category].extend(filtered_matches)
        
        # Убираем дубликаты и пустые значения
        for category in facts:
            facts[category] = list(set([f for f in facts[category] if f]))
        
        return facts
    
    def extract_facts_from_dialog(self, dialog: Dict) -> Dict[str, Any]:
        """Извлекает факты из диалога"""
        facts = {
            'dialog_id': dialog.get('dialog_id', 'unknown'),
            'user_messages': [],
            'extracted_facts': {}
        }
        
        # Собираем все сообщения пользователя
        for session in dialog.get('sessions', []):
            for message in session.get('messages', []):
                if message.get('role') == 'user':
                    user_text = message.get('content', '')
                    if user_text:
                        facts['user_messages'].append(user_text)
        
        # Объединяем все сообщения пользователя
        all_user_text = ' '.join(facts['user_messages'])
        
        # Извлекаем факты
        facts['extracted_facts'] = self.extract_facts_from_text(all_user_text)
        
        return facts
    
    def create_prompt_for_dialog(self, dialog_facts: Dict[str, Any]) -> str:
        """Создает промпт для работы с фактами диалога"""
        dialog_id = dialog_facts['dialog_id']
        facts = dialog_facts['extracted_facts']
        
        prompt = f"""# Промпт для работы с фактами диалога {dialog_id}

## Извлеченные факты о пользователе:

"""
        
        for category, values in facts.items():
            if values:
                category_name = {
                    'name': 'Имя',
                    'age': 'Возраст', 
                    'profession': 'Профессия',
                    'location': 'Местоположение',
                    'family': 'Семья',
                    'hobbies': 'Увлечения',
                    'pets': 'Питомцы'
                }.get(category, category)
                
                prompt += f"**{category_name}:** {', '.join(values)}\n"
        
        prompt += f"""

## Сообщения пользователя:
{chr(10).join(dialog_facts['user_messages'][:3])}...

## Инструкции для работы с фактами:
1. Используйте извлеченные факты для персонализированных ответов
2. Помните контекст диалога при общении с пользователем
3. Обновляйте факты при получении новой информации
4. Проверяйте достоверность фактов при необходимости

---
*Промпт создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return prompt

def load_dialogs_from_jsonl(file_path: str) -> List[Dict]:
    """Загружает диалоги из JSONL файла"""
    dialogs = []
    
    print(f"Загружаем данные из файла: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    dialogs.append(data)
                    if line_num % 100 == 0:
                        print(f"Обработано строк: {line_num}")
                except json.JSONDecodeError as e:
                    print(f"Ошибка при загрузке строки {line_num}: {e}")
                    continue
    
    return dialogs

def main():
    """Основная функция"""
    print("🎯 Извлечение реальных фактов из диалогов")
    print("="*60)
    
    # Путь к файлу данных
    data_file = "data/format_example.jsonl"
    
    if not Path(data_file).exists():
        print(f"❌ Файл {data_file} не найден!")
        return
    
    try:
        # Загружаем диалоги
        print("📂 Загружаем диалоги...")
        dialogs = load_dialogs_from_jsonl(data_file)
        print(f"✅ Загружено {len(dialogs)} диалогов")
        
        # Создаем извлекатель фактов
        extractor = RealFactExtractor()
        
        # Создаем папку для результатов
        output_dir = Path("real_facts_prompts")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🔧 Извлекаем факты из диалогов...")
        
        # Обрабатываем каждый диалог
        for i, dialog in enumerate(dialogs, 1):
            print(f"  • Обрабатываем диалог {i}...")
            
            # Извлекаем факты
            dialog_facts = extractor.extract_facts_from_dialog(dialog)
            
            # Создаем промпт
            prompt = extractor.create_prompt_for_dialog(dialog_facts)
            
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
