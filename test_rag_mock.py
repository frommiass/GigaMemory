#!/usr/bin/env python3
"""
Тест RAG интеграции с моком ModelInference
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

# Мокаем ModelInference
class MockModelInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
    
    def inference(self, messages):
        # Простой мок - возвращаем фиксированный ответ
        return "Это тестовый ответ от мока модели."

# Мокаем llm_inference модуль
sys.modules['submit.llm_inference'] = type('llm_inference', (), {'ModelInference': MockModelInference})()

def test_rag_integration_with_mock():
    """Тестирует RAG интеграцию с моком"""
    print("🧪 Тестирование RAG интеграции с моком...")
    
    try:
        # Импортируем основную модель
        from submit.model_inference import SubmitModelWithMemory
        
        # Создаем экземпляр модели
        print("\n1️⃣ Создание модели с RAG системой:")
        model = SubmitModelWithMemory("/fake/model/path")
        print("   ✅ Модель создана успешно!")
        print(f"   RAG интерфейс: {type(model.rag_interface).__name__}")
        print(f"   ModelInference: {type(model.model_inference).__name__}")
        
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
        model.write_to_memory(test_messages, dialogue_id)
        print("   ✅ Сообщения записаны в память!")
        
        # Проверяем статистику памяти
        memory_stats = model.storage.get_memory_stats(dialogue_id)
        print(f"   Сообщений в памяти: {memory_stats['messages_count']}")
        print(f"   Сессий: {memory_stats['sessions_count']}")
        
        # Тест 3: Анализ вопросов
        print("\n3️⃣ Тестирование анализа вопросов:")
        test_questions = [
            "Как меня зовут?",
            "Какие у меня животные?",
            "Где я работаю?",
            "Что я делаю в свободное время?"
        ]
        
        for question in test_questions:
            analysis = model.analyze_question(dialogue_id, question)
            if 'error' not in analysis:
                print(f"   '{question}' → Тема: {analysis.get('topic', 'N/A')}, Стратегия: {analysis.get('strategy', 'N/A')}")
            else:
                print(f"   '{question}' → Ошибка: {analysis['error']}")
        
        # Тест 4: Генерация промптов
        print("\n4️⃣ Тестирование генерации промптов:")
        for question in test_questions[:2]:
            prompt = model.answer_to_question_mock(dialogue_id, question)
            print(f"   Вопрос: '{question}'")
            print(f"   Промпт: {prompt[:150]}...")
            print(f"   Длина промпта: {len(prompt)}")
        
        # Тест 5: Полный ответ (с моком)
        print("\n5️⃣ Тестирование полного ответа:")
        for question in test_questions[:2]:
            answer = model.answer_to_question(dialogue_id, question)
            print(f"   Вопрос: '{question}'")
            print(f"   Ответ: {answer}")
        
        # Тест 6: RAG статистика
        print("\n6️⃣ Тестирование RAG статистики:")
        rag_stats = model.get_rag_stats(dialogue_id)
        if 'error' not in rag_stats:
            print(f"   Диалог: {rag_stats['dialogue_id']}")
            print(f"   Сообщений в памяти: {rag_stats['memory_messages']}")
            print(f"   Сессий: {rag_stats['sessions_count']}")
            print(f"   Доступных тем: {len(rag_stats['available_topics'])}")
            print(f"   Порог уверенности: {rag_stats['rag_config'].get('confidence_threshold', 'N/A')}")
        else:
            print(f"   Ошибка: {rag_stats['error']}")
        
        # Тест 7: Очистка памяти
        print("\n7️⃣ Тестирование очистки памяти:")
        model.clear_memory(dialogue_id)
        print("   ✅ Память очищена!")
        
        memory_after_clear = model.storage.get_memory(dialogue_id)
        print(f"   Сообщений после очистки: {len(memory_after_clear)}")
        
        print("\n✅ RAG интеграция с моком протестирована успешно!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rag_integration_with_mock()
