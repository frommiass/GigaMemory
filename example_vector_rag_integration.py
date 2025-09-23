#!/usr/bin/env python3
"""
Пример интеграции векторного RAG в GigaMemory
Показывает как заменить обычный RAG на векторный
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
from src.submit.model_inference_vector import SubmitModelWithVectorMemory


def example_basic_integration():
    """Пример базовой интеграции"""
    print("🚀 Пример базовой интеграции векторного RAG")
    print("=" * 50)
    
    # Создаем модель с векторным RAG
    model = SubmitModelWithVectorMemory(
        model_path="path/to/your/model",  # Замените на реальный путь
        use_vector_rag=True,
        vector_model="cointegrated/rubert-tiny2",
        use_gpu=True,
        enable_hybrid_search=True
    )
    
    print("✅ Модель с векторным RAG создана")
    
    # Пример диалога
    dialogue_id = "example_dialogue"
    messages = [
        Message(role="user", content="Привет! Меня зовут Алексей, я из Москвы."),
        Message(role="assistant", content="Здравствуйте, Алексей! Рад познакомиться."),
        Message(role="user", content="Я работаю программистом в Яндексе."),
        Message(role="assistant", content="Отлично! Яндекс - крупная IT-компания."),
        Message(role="user", content="У меня есть кот по имени Барсик."),
        Message(role="assistant", content="Барсик - прекрасное имя для кота!"),
    ]
    
    # Сохраняем сообщения в память (автоматическая векторизация)
    print("\n📝 Сохранение сообщений в память...")
    model.write_to_memory(messages, dialogue_id)
    print("✅ Сообщения сохранены и проиндексированы")
    
    # Тестовые вопросы
    test_questions = [
        "Как меня зовут?",
        "Где я работаю?",
        "Как зовут моего кота?",
        "Откуда я?"
    ]
    
    print("\n❓ Тестирование вопросов:")
    for question in test_questions:
        print(f"\nВопрос: {question}")
        
        # Получаем результаты векторного поиска для анализа
        search_results = model.get_vector_search_results(question, dialogue_id, top_k=3)
        
        if search_results:
            print("🔍 Найденные релевантные фрагменты:")
            for i, result in enumerate(search_results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Text: {result['text'][:100]}...")
        
        # Генерируем ответ (в реальном использовании)
        # answer = model.answer_to_question(dialogue_id, question)
        # print(f"Ответ: {answer}")
    
    # Статистика
    print("\n📊 Статистика векторного RAG:")
    stats = model.get_vector_rag_stats()
    print(f"  Векторный RAG включен: {stats.get('vector_rag_enabled', False)}")
    print(f"  Всего диалогов: {stats.get('total_dialogues', 0)}")
    print(f"  Всего хранилищ: {stats.get('total_stores', 0)}")
    
    if 'embedding_stats' in stats:
        emb_stats = stats['embedding_stats']
        print(f"  Закодировано текстов: {emb_stats.get('total_encoded', 0)}")
        print(f"  Cache hit rate: {emb_stats.get('cache_hit_rate', 0):.1%}")
    
    # Сохранение индексов
    print("\n💾 Сохранение векторных индексов...")
    model.save_vector_indices("./example_vector_indices")
    print("✅ Индексы сохранены")
    
    return model


def example_advanced_integration():
    """Пример продвинутой интеграции с настройками"""
    print("\n🔧 Пример продвинутой интеграции")
    print("=" * 50)
    
    # Создаем модель с продвинутыми настройками
    model = SubmitModelWithVectorMemory(
        model_path="path/to/your/model",
        use_vector_rag=True,
        vector_model="cointegrated/LaBSE-en-ru",  # Более качественная модель
        use_gpu=True,
        enable_hybrid_search=True
    )
    
    print("✅ Модель с продвинутыми настройками создана")
    
    # Загружаем сохраненные индексы
    print("\n📂 Загрузка сохраненных индексов...")
    model.load_vector_indices("./example_vector_indices")
    print("✅ Индексы загружены")
    
    # Тестируем поиск
    dialogue_id = "example_dialogue"
    question = "информация о работе и питомцах"
    
    print(f"\n🔍 Расширенный поиск: '{question}'")
    results = model.get_vector_search_results(question, dialogue_id, top_k=5)
    
    print(f"Найдено {len(results)} результатов:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Релевантность: {result['score']:.3f}")
        print(f"   Текст: {result['text']}")
        print(f"   Метаданные: {result['metadata']}")
    
    return model


def example_comparison():
    """Пример сравнения векторного и традиционного RAG"""
    print("\n⚖️ Сравнение векторного и традиционного RAG")
    print("=" * 50)
    
    # Модель с векторным RAG
    vector_model = SubmitModelWithVectorMemory(
        model_path="path/to/your/model",
        use_vector_rag=True,
        vector_model="cointegrated/rubert-tiny2"
    )
    
    # Модель с традиционным RAG
    traditional_model = SubmitModelWithVectorMemory(
        model_path="path/to/your/model",
        use_vector_rag=False
    )
    
    print("✅ Созданы модели для сравнения")
    
    # Подготавливаем тестовые данные
    dialogue_id = "comparison_test"
    messages = [
        Message(role="user", content="Меня зовут Мария, я врач-кардиолог."),
        Message(role="assistant", content="Здравствуйте, Мария! Кардиология - важная область медицины."),
        Message(role="user", content="Я работаю в больнице имени Боткина."),
        Message(role="assistant", content="Больница Боткина - известное медицинское учреждение."),
        Message(role="user", content="У меня есть дочь Анна, ей 8 лет."),
        Message(role="assistant", content="Анна - прекрасное имя для дочери!"),
    ]
    
    # Индексируем данные
    print("\n📝 Индексация данных...")
    vector_model.write_to_memory(messages, dialogue_id)
    traditional_model.write_to_memory(messages, dialogue_id)
    print("✅ Данные проиндексированы")
    
    # Тестовые вопросы
    test_questions = [
        "Как зовут врача?",
        "Где работает Мария?",
        "Как зовут дочь?",
        "Сколько лет дочери?"
    ]
    
    print("\n❓ Сравнение результатов:")
    for question in test_questions:
        print(f"\nВопрос: {question}")
        
        # Векторный поиск
        vector_results = vector_model.get_vector_search_results(question, dialogue_id, top_k=2)
        print("🔍 Векторный RAG:")
        for result in vector_results:
            print(f"  Score: {result['score']:.3f} - {result['text'][:60]}...")
        
        # Традиционный RAG (симуляция)
        print("📚 Традиционный RAG:")
        print("  Использует keyword matching и классификацию")
        print("  Менее точный для семантического поиска")


def example_migration_guide():
    """Руководство по миграции с обычного RAG на векторный"""
    print("\n📋 Руководство по миграции")
    print("=" * 50)
    
    migration_code = '''
# ============== МИГРАЦИЯ С ОБЫЧНОГО RAG НА ВЕКТОРНЫЙ ==============

# БЫЛО (в src/submit/model_inference.py):
from .rag.engine import RAGEngine

class SubmitModelWithMemory(ModelWithMemory):
    def __init__(self, model_path: str) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        self.rag_interface = RAGEngine()  # Обычный RAG

# СТАЛО (замените на):
from .rag.vector_rag_interface import VectorRAGInterface

class SubmitModelWithMemory(ModelWithMemory):
    def __init__(self, model_path: str) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        self.rag_interface = VectorRAGInterface(  # Векторный RAG
            model_name="cointegrated/rubert-tiny2",
            use_gpu=True,
            enable_hybrid_search=True
        )

# ============== АЛЬТЕРНАТИВНЫЙ СПОСОБ ==============

# Или используйте готовый класс:
from .model_inference_vector import SubmitModelWithVectorMemory

# Просто замените класс:
class SubmitModelWithMemory(SubmitModelWithVectorMemory):
    def __init__(self, model_path: str) -> None:
        super().__init__(
            model_path=model_path,
            use_vector_rag=True,  # Включить векторный RAG
            vector_model="cointegrated/rubert-tiny2",
            use_gpu=True,
            enable_hybrid_search=True
        )

# ============== НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ ==============

# Для больших объемов данных:
config = VectorRAGConfig(
    embedding_batch_size=64,      # Больше батч = быстрее
    use_amp=True,                # Mixed precision на GPU
    compile_model=True,          # torch.compile (PyTorch 2.0+)
    optimize_on_save=True       # Оптимизация при сохранении
)

# ============== МОНИТОРИНГ ==============

# Получение статистики:
stats = model.get_vector_rag_stats()
print(f"Cache hit rate: {stats['embedding_stats']['cache_hit_rate']:.1%}")
print(f"Всего диалогов: {stats['total_dialogues']}")

# Сохранение/загрузка индексов:
model.save_vector_indices("./my_indices")
model.load_vector_indices("./my_indices")
'''
    
    print(migration_code)


def main():
    """Главная функция с примерами"""
    print("🎯 ПРИМЕРЫ ИНТЕГРАЦИИ ВЕКТОРНОГО RAG В GIGAMEMORY")
    print("=" * 60)
    
    try:
        # Базовый пример
        example_basic_integration()
        
        # Продвинутый пример
        example_advanced_integration()
        
        # Сравнение
        example_comparison()
        
        # Руководство по миграции
        example_migration_guide()
        
        print("\n🎉 Все примеры выполнены успешно!")
        print("VectorRAGInterface готов к использованию в вашем проекте!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
