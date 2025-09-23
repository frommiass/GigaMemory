#!/usr/bin/env python3
"""
Демонстрация генерации промптов в новой архитектуре
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
import logging
import time
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptGenerator:
    """Генератор промптов для демонстрации"""
    
    def __init__(self):
        self.prompt_templates = {
            'work': "Расскажи о профессиональной деятельности пользователя: {context}",
            'family': "Опиши семейную ситуацию пользователя: {context}",
            'location': "Где живет пользователь: {context}",
            'hobby': "Какие увлечения у пользователя: {context}",
            'pets': "Расскажи о домашних животных пользователя: {context}",
            'education': "Что изучает или изучал пользователь: {context}",
            'general': "Используя информацию из памяти, ответь на вопрос: {question}\n\nКонтекст: {context}"
        }
        
        self.fact_categories = {
            'work': ['программист', 'работа', 'компания', 'яндекс', 'python', 'разработка'],
            'family': ['жена', 'дочь', 'семья', 'мария', 'анна', 'ребенок'],
            'location': ['москва', 'живет', 'сокольники', 'район', 'адрес'],
            'hobby': ['фотографией', 'увлекается', 'хобби', 'рисовать', 'выходные'],
            'pets': ['собака', 'рекс', 'животные', 'питомец'],
            'education': ['изучает', 'обучение', 'нейросети', 'машинное обучение']
        }
    
    def classify_question(self, question: str) -> str:
        """Классифицирует вопрос по категориям"""
        question_lower = question.lower()
        
        for category, keywords in self.fact_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def extract_context_from_memory(self, memory_data: List[str], question: str) -> str:
        """Извлекает релевантный контекст из памяти"""
        question_lower = question.lower()
        relevant_contexts = []
        
        for memory_item in memory_data:
            memory_lower = memory_item.lower()
            
            # Простая проверка релевантности по ключевым словам
            question_words = set(question_lower.split())
            memory_words = set(memory_lower.split())
            
            # Если есть пересечение слов, считаем релевантным
            if question_words.intersection(memory_words):
                relevant_contexts.append(memory_item)
        
        return "\n".join(relevant_contexts[:3])  # Берем топ-3 релевантных
    
    def generate_prompt(self, question: str, memory_data: List[str]) -> str:
        """Генерирует промпт для модели"""
        # Классифицируем вопрос
        category = self.classify_question(question)
        
        # Извлекаем релевантный контекст
        context = self.extract_context_from_memory(memory_data, question)
        
        # Выбираем шаблон
        if category == 'general':
            template = self.prompt_templates['general']
            return template.format(question=question, context=context)
        else:
            template = self.prompt_templates[category]
            return template.format(context=context)

class AdvancedPromptGenerator(PromptGenerator):
    """Продвинутый генератор промптов с фактами"""
    
    def __init__(self):
        super().__init__()
        self.extracted_facts = {
            'work': [],
            'family': [],
            'location': [],
            'hobby': [],
            'pets': [],
            'education': []
        }
    
    def extract_facts_from_text(self, text: str) -> Dict[str, List[str]]:
        """Извлекает факты из текста"""
        facts = {category: [] for category in self.fact_categories.keys()}
        
        text_lower = text.lower()
        
        # Извлечение фактов о работе
        if 'программист' in text_lower:
            facts['work'].append('Работает программистом')
        if 'яндекс' in text_lower:
            facts['work'].append('Работает в компании Яндекс')
        if 'python' in text_lower:
            facts['work'].append('Работает с Python')
        
        # Извлечение фактов о семье
        if 'жена' in text_lower:
            facts['family'].append('Есть жена')
        if 'дочь' in text_lower:
            facts['family'].append('Есть дочь')
        if 'анна' in text_lower and '5 лет' in text_lower:
            facts['family'].append('Дочери 5 лет')
        
        # Извлечение фактов о местоположении
        if 'москва' in text_lower:
            facts['location'].append('Живет в Москве')
        if 'сокольники' in text_lower:
            facts['location'].append('Живет в районе Сокольники')
        
        # Извлечение фактов об увлечениях
        if 'фотографией' in text_lower:
            facts['hobby'].append('Увлекается фотографией')
        
        # Извлечение фактов о животных
        if 'собака' in text_lower and 'рекс' in text_lower:
            facts['pets'].append('Есть собака по кличке Рекс')
        
        # Извлечение фактов об образовании
        if 'изучает' in text_lower and 'нейросети' in text_lower:
            facts['education'].append('Изучает нейросети')
        
        return facts
    
    def update_facts(self, new_facts: Dict[str, List[str]]):
        """Обновляет базу фактов"""
        for category, fact_list in new_facts.items():
            self.extracted_facts[category].extend(fact_list)
            # Убираем дубликаты
            self.extracted_facts[category] = list(set(self.extracted_facts[category]))
    
    def generate_enhanced_prompt(self, question: str, memory_data: List[str]) -> str:
        """Генерирует улучшенный промпт с фактами"""
        # Классифицируем вопрос
        category = self.classify_question(question)
        
        # Извлекаем релевантный контекст
        context = self.extract_context_from_memory(memory_data, question)
        
        # Получаем релевантные факты
        relevant_facts = []
        if category in self.extracted_facts:
            relevant_facts = self.extracted_facts[category]
        
        # Добавляем факты из других категорий, если они могут быть полезны
        if category == 'general':
            for cat_facts in self.extracted_facts.values():
                relevant_facts.extend(cat_facts)
        
        # Формируем промпт
        prompt_parts = [
            "Ты - помощник с доступом к структурированной памяти пользователя.",
            "Используй следующую информацию для точного ответа на вопрос:",
            "",
            "ВОПРОС: " + question,
            ""
        ]
        
        if relevant_facts:
            prompt_parts.append("ИЗВЛЕЧЕННЫЕ ФАКТЫ:")
            for fact in relevant_facts[:5]:  # Ограничиваем количество фактов
                prompt_parts.append(f"• {fact}")
            prompt_parts.append("")
        
        if context:
            prompt_parts.extend([
                "КОНТЕКСТ ИЗ ПАМЯТИ:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "ИНСТРУКЦИИ:",
            "- Отвечай кратко и по существу",
            "- Используй только информацию из предоставленного контекста",
            "- Если информации недостаточно, скажи об этом",
            "- Будь точным в деталях"
        ])
        
        return "\n".join(prompt_parts)

def demo_prompt_generation():
    """Демонстрирует генерацию промптов"""
    
    print("🎯 Демонстрация генерации промптов в новой архитектуре")
    print("=" * 70)
    
    # Тестовые данные
    memory_data = [
        "Привет! Меня зовут Алексей, я работаю программистом в компании Яндекс.",
        "У меня есть жена Мария и дочь Анна, которой 5 лет.",
        "Я живу в Москве, в районе Сокольники. Работаю удаленно.",
        "Моя жена работает учителем в школе. Она очень любит свою работу.",
        "У нас есть собака по кличке Рекс. Он очень дружелюбный.",
        "Я увлекаюсь фотографией и часто фотографирую семью на выходных.",
        "Наша дочь ходит в детский сад и очень любит рисовать.",
        "Я работаю с Python и машинным обучением. Недавно изучаю нейросети."
    ]
    
    test_questions = [
        "Расскажи о работе пользователя",
        "Какая у пользователя семья?",
        "Где живет пользователь?",
        "Какие у пользователя увлечения?",
        "Расскажи о домашних животных",
        "Что изучает пользователь?",
        "Где работает жена пользователя?",
        "Сколько лет дочери пользователя?",
        "Расскажи все, что знаешь о пользователе"
    ]
    
    # 1. Базовый генератор промптов
    print("\n📝 1. Базовый генератор промптов:")
    print("-" * 50)
    
    basic_generator = PromptGenerator()
    
    for i, question in enumerate(test_questions[:3], 1):
        print(f"\n{i}. Вопрос: {question}")
        prompt = basic_generator.generate_prompt(question, memory_data)
        print("Сгенерированный промпт:")
        print(prompt)
        print("-" * 30)
    
    # 2. Продвинутый генератор с фактами
    print("\n📊 2. Продвинутый генератор с извлечением фактов:")
    print("-" * 50)
    
    advanced_generator = AdvancedPromptGenerator()
    
    # Извлекаем факты из всех сообщений
    print("Извлечение фактов из памяти...")
    for memory_item in memory_data:
        facts = advanced_generator.extract_facts_from_text(memory_item)
        advanced_generator.update_facts(facts)
    
    print("Извлеченные факты:")
    for category, facts in advanced_generator.extracted_facts.items():
        if facts:
            print(f"  {category.upper()}:")
            for fact in facts:
                print(f"    • {fact}")
    
    # Генерируем улучшенные промпты
    print(f"\n🤖 Генерация улучшенных промптов:")
    print("-" * 50)
    
    for i, question in enumerate(test_questions[:5], 1):
        print(f"\n{i}. Вопрос: {question}")
        enhanced_prompt = advanced_generator.generate_enhanced_prompt(question, memory_data)
        print("Улучшенный промпт:")
        print(enhanced_prompt)
        print("-" * 30)
    
    # 3. Сравнение подходов
    print(f"\n⚖️ 3. Сравнение подходов:")
    print("-" * 50)
    
    sample_question = "Расскажи о семье пользователя"
    
    print(f"Вопрос: {sample_question}")
    print("\nБазовый промпт:")
    basic_prompt = basic_generator.generate_prompt(sample_question, memory_data)
    print(basic_prompt)
    
    print("\nУлучшенный промпт:")
    enhanced_prompt = advanced_generator.generate_enhanced_prompt(sample_question, memory_data)
    print(enhanced_prompt)
    
    print(f"\n🎉 Демонстрация завершена!")
    print(f"✅ Показаны различные подходы к генерации промптов")
    print(f"✅ Продемонстрировано извлечение фактов")
    print(f"✅ Сравнены базовый и улучшенный подходы")

if __name__ == "__main__":
    demo_prompt_generation()
