#!/usr/bin/env python3
"""
Тестирование оптимизированной системы SmartMemory
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
from submit.smart_memory_optimized import OptimizedSmartMemory
from submit.config_loader import config_manager
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_optimized_system():
    """Тестирует оптимизированную систему"""
    
    print("🚀 Запуск тестирования оптимизированной системы SmartMemory")
    print("=" * 60)
    
    try:
        # 1. Инициализация системы
        print("📋 Инициализация системы...")
        memory = OptimizedSmartMemory()
        print("✅ Система инициализирована")
        
        # 2. Тестовые сообщения
        print("\n📝 Подготовка тестовых данных...")
        test_messages = [
            Message(role="user", content="Привет! Меня зовут Алексей, я работаю программистом в компании Яндекс."),
            Message(role="user", content="У меня есть жена Мария и дочь Анна, которой 5 лет."),
            Message(role="user", content="Я живу в Москве, в районе Сокольники. Работаю удаленно."),
            Message(role="user", content="Моя жена работает учителем в школе. Она очень любит свою работу."),
            Message(role="user", content="У нас есть собака по кличке Рекс. Он очень дружелюбный."),
            Message(role="user", content="Я увлекаюсь фотографией и часто фотографирую семью на выходных."),
            Message(role="user", content="Наша дочь ходит в детский сад и очень любит рисовать."),
            Message(role="user", content="Я работаю с Python и машинным обучением. Недавно изучаю нейросети."),
        ]
        
        dialogue_id = "test_dialogue_1"
        print(f"✅ Подготовлено {len(test_messages)} тестовых сообщений")
        
        # 3. Обработка диалога
        print(f"\n🔄 Обработка диалога {dialogue_id}...")
        stats = memory.process_dialogue_optimized(dialogue_id, test_messages)
        
        print("📊 Статистика обработки:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
        
        # 4. Тестовые вопросы
        print(f"\n❓ Тестирование вопросов...")
        test_questions = [
            "Расскажи о работе пользователя",
            "Какая у пользователя семья?",
            "Где живет пользователь?",
            "Какие у пользователя увлечения?",
            "Расскажи о домашних животных",
            "Что изучает пользователь?",
            "Где работает жена пользователя?",
            "Сколько лет дочери пользователя?"
        ]
        
        # 5. Генерация ответов
        print("\n🤖 Генерация ответов:")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Вопрос: {question}")
            try:
                answer = memory.answer_question_optimized(dialogue_id, question)
                print(f"   Ответ: {answer}")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        # 6. Статистика системы
        print(f"\n📈 Общая статистика системы:")
        print("-" * 40)
        system_stats = memory.get_system_stats()
        
        for category, stats in system_stats.items():
            print(f"\n{category.upper()}:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        # 7. Сохранение состояния
        print(f"\n💾 Сохранение состояния...")
        memory.save_state("./test_state")
        print("✅ Состояние сохранено")
        
        print(f"\n🎉 Тестирование завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

def test_config_loading():
    """Тестирует загрузку конфигурации"""
    print("\n⚙️ Тестирование загрузки конфигурации...")
    
    try:
        config = config_manager.get_config()
        print("✅ Конфигурация загружена:")
        print(f"  Модель: {config.model_path}")
        print(f"  Embedding модель: {config.embedding_config.get('model_name', 'N/A')}")
        print(f"  Сжатие включено: {config.compression_config.get('enabled', False)}")
        print(f"  Кэш размер: {config.cache_config.get('max_size', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")

if __name__ == "__main__":
    # Тестируем загрузку конфигурации
    test_config_loading()
    
    # Тестируем основную систему
    test_optimized_system()
