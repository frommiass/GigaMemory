#!/usr/bin/env python3
"""
Тест полной интеграции RAG системы с основной моделью
"""
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Создаем простую модель Message
class Message:
    def __init__(self, role: str, content: str, session_id: str = None):
        self.role = role
        self.content = content
        self.session_id = session_id

# Мокаем models модуль
sys.modules['models'] = type('models', (), {'Message': Message})()

def test_full_integration():
    """Тестирует полную интеграцию RAG системы"""
    print("🧪 Тестирование полной интеграции RAG системы...")
    
    try:
        # Импортируем основную модель
        from submit.model_inference import SubmitModelWithMemory
        
        # Создаем экземпляр модели (без реального пути к модели)
        print("\n1️⃣ Создание модели с RAG системой:")
        try:
            model = SubmitModelWithMemory("/fake/model/path")
            print("   ✅ Модель создана успешно!")
            print(f"   RAG интерфейс: {type(model.rag_interface).__name__}")
        except Exception as e:
            print(f"   ❌ Ошибка создания модели: {e}")
            return
        
        # Тестовые сообщения
        test_messages = [
            Message(role="user", content="Привет, меня зовут Иван", session_id="1"),
            Message(role="assistant", content="Привет, Иван! Рад познакомиться!"),
            Message(role="user", content="У меня есть кот по имени Барсик", session_id="1"),
            Message(role="assistant", content="Как интересно! Расскажи о своем коте."),
            Message(role="user", content="Мой кот любит играть с мячиком", session_id="2"),
            Message(role="assistant", content="Звучит забавно!"),
            Message(role="user", content="Я работаю программистом в IT компании", session_id="3"),
            Message(role="assistant", content="Отлично! Какие технологии используешь?"),
        ]
        
        dialogue_id = "test_dialogue"
        
        # Тест 2: Запись в память
        print("\n2️⃣ Тестирование записи в память:")
        try:
            model.write_to_memory(test_messages, dialogue_id)
            print("   ✅ Сообщения записаны в память!")
            
            # Проверяем статистику памяти
            memory_stats = model.storage.get_memory_stats(dialogue_id)
            print(f"   Сообщений в памяти: {memory_stats['messages_count']}")
            print(f"   Сессий: {memory_stats['sessions_count']}")
        except Exception as e:
            print(f"   ❌ Ошибка записи в память: {e}")
        
        # Тест 3: Анализ вопросов
        print("\n3️⃣ Тестирование анализа вопросов:")
        test_questions = [
            "Как меня зовут?",
            "Какие у меня животные?",
            "Где я работаю?",
            "Что я делаю в свободное время?"
        ]
        
        for question in test_questions:
            try:
                analysis = model.analyze_question(dialogue_id, question)
                if 'error' not in analysis:
                    print(f"   '{question}' → Тема: {analysis.get('topic', 'N/A')}, Стратегия: {analysis.get('strategy', 'N/A')}")
                else:
                    print(f"   '{question}' → Ошибка: {analysis['error']}")
            except Exception as e:
                print(f"   '{question}' → Ошибка: {e}")
        
        # Тест 4: Генерация промптов (без LLM)
        print("\n4️⃣ Тестирование генерации промптов:")
        for question in test_questions[:2]:  # Тестируем только первые 2 вопроса
            try:
                prompt = model.answer_to_question_mock(dialogue_id, question)
                print(f"   Вопрос: '{question}'")
                print(f"   Промпт: {prompt[:100]}...")
                print(f"   Длина промпта: {len(prompt)}")
            except Exception as e:
                print(f"   Вопрос: '{question}' → Ошибка: {e}")
        
        # Тест 5: RAG статистика
        print("\n5️⃣ Тестирование RAG статистики:")
        try:
            rag_stats = model.get_rag_stats(dialogue_id)
            if 'error' not in rag_stats:
                print(f"   Диалог: {rag_stats['dialogue_id']}")
                print(f"   Сообщений в памяти: {rag_stats['memory_messages']}")
                print(f"   Сессий: {rag_stats['sessions_count']}")
                print(f"   Доступных тем: {len(rag_stats['available_topics'])}")
                print(f"   Порог уверенности: {rag_stats['rag_config'].get('confidence_threshold', 'N/A')}")
            else:
                print(f"   Ошибка: {rag_stats['error']}")
        except Exception as e:
            print(f"   Ошибка получения статистики: {e}")
        
        # Тест 6: Очистка памяти
        print("\n6️⃣ Тестирование очистки памяти:")
        try:
            model.clear_memory(dialogue_id)
            print("   ✅ Память очищена!")
            
            # Проверяем, что память пуста
            memory_after_clear = model.storage.get_memory(dialogue_id)
            print(f"   Сообщений после очистки: {len(memory_after_clear)}")
        except Exception as e:
            print(f"   ❌ Ошибка очистки памяти: {e}")
        
        print("\n✅ Полная интеграция протестирована!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_integration()
