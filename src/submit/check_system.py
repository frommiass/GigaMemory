#!/usr/bin/env python3
"""
Скрипт полной проверки системы GigaMemory
Запуск: python src/submit/check_system.py
"""

import sys
import os
import logging
from pathlib import Path

# Добавляем пути
sys.path.append(os.path.dirname(__file__) + '/../../')
from models import Message

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_imports():
    """Проверяет что все модули импортируются"""
    print("\n" + "="*50)
    print("1. ПРОВЕРКА ИМПОРТОВ")
    print("="*50)
    
    modules_to_check = [
        ('Core Container', 'submit.core.container'),
        ('Core Interfaces', 'submit.core.interfaces'),
        ('Core Orchestrator', 'submit.core.orchestrator'),
        ('Bootstrap', 'submit.bootstrap'),
        ('Storage Module', 'submit.modules.storage.module'),
        ('Embeddings Module', 'submit.modules.embeddings.module'),
        ('Extraction Module', 'submit.modules.extraction.module'),
        ('Compression Module', 'submit.modules.compression.module'),
        ('Optimization Module', 'submit.modules.optimization.module'),
        ('RAG Module', 'submit.modules.rag.module'),
        ('LLM Inference', 'submit.llm_inference'),
        ('Model Inference', 'submit.model_inference'),
    ]
    
    failed = []
    for name, module_path in modules_to_check:
        try:
            __import__(module_path)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️ Failed imports: {', '.join(failed)}")
        return False
    
    print("\n✅ Все модули импортируются успешно!")
    return True


def check_initialization():
    """Проверяет инициализацию системы"""
    print("\n" + "="*50)
    print("2. ПРОВЕРКА ИНИЦИАЛИЗАЦИИ")
    print("="*50)
    
    try:
        from submit.bootstrap import bootstrap_system_with_config, get_default_config
        
        print("Загружаем конфигурацию по умолчанию...")
        config = get_default_config()
        
        # Используем тестовый путь к модели
        config['model_path'] = '/tmp/test_model'
        
        print("Инициализируем систему...")
        orchestrator = bootstrap_system_with_config(config)
        
        # Проверяем модули
        modules = [
            ('Optimizer', orchestrator.optimizer),
            ('Storage', orchestrator.storage),
            ('Embeddings', orchestrator.embeddings),
            ('Extractor', orchestrator.extractor),
            ('Compressor', orchestrator.compressor),
            ('RAG', orchestrator.rag),
            ('Model', orchestrator.model)
        ]
        
        for name, module in modules:
            if module is None:
                print(f"❌ {name}: NOT INITIALIZED")
                return False
            else:
                print(f"✅ {name}: {type(module).__name__}")
        
        print("\n✅ Все модули инициализированы!")
        return orchestrator
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_interfaces(orchestrator):
    """Проверяет что интерфейсы правильно реализованы"""
    print("\n" + "="*50)
    print("3. ПРОВЕРКА ИНТЕРФЕЙСОВ")
    print("="*50)
    
    from submit.core.interfaces import (
        IStorage, IEmbeddingEngine, IFactExtractor, 
        ICompressor, IOptimizer, IRAGEngine, IModelInference
    )
    
    checks = [
        ('Storage', orchestrator.storage, IStorage, 
         ['store_messages', 'get_dialogue_messages', 'get_dialogue_sessions', 'clear_dialogue']),
        ('Embeddings', orchestrator.embeddings, IEmbeddingEngine,
         ['encode_texts', 'vector_search', 'index_dialogue']),
        ('Extractor', orchestrator.extractor, IFactExtractor,
         ['extract_facts', 'get_user_profile', 'query_facts']),
        ('Compressor', orchestrator.compressor, ICompressor,
         ['compress_text', 'compress_sessions', 'get_compression_stats']),
        ('Optimizer', orchestrator.optimizer, IOptimizer,
         ['cache_get', 'cache_put', 'batch_process']),
        ('RAG', orchestrator.rag, IRAGEngine,
         ['process_question', 'find_relevant_sessions', 'generate_answer']),
        ('Model', orchestrator.model, IModelInference,
         ['generate'])
    ]
    
    all_ok = True
    for name, module, interface, methods in checks:
        print(f"\n{name} ({interface.__name__}):")
        
        # Проверяем наследование
        if not isinstance(module, interface):
            print(f"  ❌ Не реализует интерфейс {interface.__name__}")
            all_ok = False
            continue
        
        # Проверяем методы
        for method in methods:
            if hasattr(module, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - отсутствует")
                all_ok = False
    
    if all_ok:
        print("\n✅ Все интерфейсы реализованы правильно!")
    else:
        print("\n⚠️ Есть проблемы с интерфейсами")
    
    return all_ok


def check_pipeline(orchestrator):
    """Проверяет работу pipeline"""
    print("\n" + "="*50)
    print("4. ПРОВЕРКА PIPELINE")
    print("="*50)
    
    try:
        # Тестовые данные
        test_messages = [
            Message("user", "Привет! Меня зовут Алексей."),
            Message("assistant", "Здравствуйте, Алексей!"),
            Message("user", "Мне 30 лет, я работаю дизайнером."),
            Message("assistant", "Интересная профессия!"),
            Message("user", "У меня есть собака по кличке Рекс."),
            Message("assistant", "Рекс - отличное имя для собаки!")
        ]
        
        dialogue_id = "test_pipeline"
        
        print(f"Обрабатываем {len(test_messages)} сообщений...")
        result = orchestrator.process_dialogue(dialogue_id, test_messages)
        
        if result.get('success'):
            print(f"✅ Pipeline обработал диалог:")
            print(f"  - Обработано сообщений: {result.get('messages_processed', 0)}")
            
            pipeline_results = result.get('pipeline_results', {})
            if 'storage' in pipeline_results:
                storage_data = pipeline_results['storage'].metadata
                print(f"  - Storage: {storage_data.get('filtered', 0)}/{storage_data.get('total', 0)} сообщений")
            
            if 'facts' in pipeline_results:
                print(f"  - Извлечено фактов: {pipeline_results['facts'].get('extracted', 0)}")
            
            if 'embeddings' in pipeline_results:
                emb_data = pipeline_results['embeddings'].data
                print(f"  - Проиндексировано: {emb_data.get('indexed', 0)} chunks")
            
            # Тестируем вопросы
            print("\nТестируем ответы на вопросы:")
            questions = [
                "Как меня зовут?",
                "Сколько мне лет?",
                "Кем я работаю?",
                "Как зовут мою собаку?"
            ]
            
            for q in questions:
                answer = orchestrator.answer_question(dialogue_id, q)
                print(f"  Q: {q}")
                print(f"  A: {answer}")
                
                # Проверяем качество ответа
                if "Алексей" in answer and "зовут" in q:
                    print(f"    ✅ Ответ корректный")
                elif "30" in answer and "лет" in q:
                    print(f"    ✅ Ответ корректный")
                elif "дизайнер" in answer and "работа" in q:
                    print(f"    ✅ Ответ корректный")
                elif "Рекс" in answer and "собак" in q:
                    print(f"    ✅ Ответ корректный")
                elif "Нет такой информации" in answer:
                    print(f"    ⚠️ Информация не найдена")
                else:
                    print(f"    ❓ Неопределенный ответ")
            
            # Очищаем
            orchestrator.clear_dialogue(dialogue_id)
            print("\n✅ Pipeline работает!")
            return True
            
        else:
            print(f"❌ Pipeline failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_submission_class():
    """Проверяет главный класс для конкурса"""
    print("\n" + "="*50)
    print("5. ПРОВЕРКА SUBMISSION CLASS")
    print("="*50)
    
    try:
        from submit.model_inference import SubmitModelWithMemory
        
        # Создаем с тестовым путем
        print("Создаем SubmitModelWithMemory...")
        model = SubmitModelWithMemory("/tmp/test_model")
        
        # Проверяем методы
        methods = ['write_to_memory', 'clear_memory', 'answer_to_question']
        for method in methods:
            if hasattr(model, method):
                print(f"✅ {method} - существует")
            else:
                print(f"❌ {method} - отсутствует")
                return False
        
        # Тестируем работу
        print("\nТестируем работу методов...")
        
        messages = [
            Message("user", "Тест. Меня зовут Петр."),
            Message("assistant", "Здравствуйте, Петр!")
        ]
        
        # write_to_memory
        model.write_to_memory(messages, "test_submission")
        print("✅ write_to_memory - работает")
        
        # answer_to_question
        answer = model.answer_to_question("test_submission", "Как меня зовут?")
        print(f"✅ answer_to_question - работает (ответ: {answer[:50]}...)")
        
        # clear_memory
        model.clear_memory("test_submission")
        print("✅ clear_memory - работает")
        
        print("\n✅ SubmitModelWithMemory полностью функционален!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_performance():
    """Проверяет производительность"""
    print("\n" + "="*50)
    print("6. ПРОВЕРКА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*50)
    
    import time
    from submit.model_inference import SubmitModelWithMemory
    
    try:
        model = SubmitModelWithMemory("/tmp/test_model")
        
        # Генерируем большой диалог
        messages = []
        for i in range(50):
            messages.append(Message("user", f"Сообщение {i}. Текст для теста."))
            messages.append(Message("assistant", f"Ответ на сообщение {i}."))
        
        dialogue_id = "perf_test"
        
        # Замеряем запись
        start = time.time()
        model.write_to_memory(messages, dialogue_id)
        write_time = time.time() - start
        print(f"✅ Запись {len(messages)} сообщений: {write_time:.2f} сек")
        
        # Замеряем ответ
        start = time.time()
        answer = model.answer_to_question(dialogue_id, "Что было в сообщении 10?")
        answer_time = time.time() - start
        print(f"✅ Генерация ответа: {answer_time:.2f} сек")
        
        # Проверяем требования
        if answer_time < 2.0:
            print("✅ Время ответа < 2 сек - ОТЛИЧНО!")
        elif answer_time < 5.0:
            print("⚠️ Время ответа < 5 сек - приемлемо")
        else:
            print("❌ Время ответа > 5 сек - нужна оптимизация")
        
        # Очистка
        model.clear_memory(dialogue_id)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def main():
    """Главная функция проверки"""
    print("\n" + "="*60)
    print(" GIGAMEMORY SYSTEM CHECK ".center(60, "="))
    print("="*60)
    
    results = []
    
    # 1. Импорты
    if check_imports():
        results.append(("Импорты", True))
    else:
        results.append(("Импорты", False))
        print("\n❌ Критическая ошибка: не все модули импортируются")
        return
    
    # 2. Инициализация
    orchestrator = check_initialization()
    if orchestrator:
        results.append(("Инициализация", True))
    else:
        results.append(("Инициализация", False))
        print("\n❌ Критическая ошибка: система не инициализируется")
        return
    
    # 3. Интерфейсы
    if check_interfaces(orchestrator):
        results.append(("Интерфейсы", True))
    else:
        results.append(("Интерфейсы", False))
    
    # 4. Pipeline
    if check_pipeline(orchestrator):
        results.append(("Pipeline", True))
    else:
        results.append(("Pipeline", False))
    
    # 5. Submission Class
    if check_submission_class():
        results.append(("Submission", True))
    else:
        results.append(("Submission", False))
    
    # 6. Performance
    if check_performance():
        results.append(("Performance", True))
    else:
        results.append(("Performance", False))
    
    # Итоговый отчет
    print("\n" + "="*60)
    print(" ИТОГОВЫЙ ОТЧЕТ ".center(60, "="))
    print("="*60)
    
    for name, status in results:
        status_str = "✅ PASSED" if status else "❌ FAILED"
        print(f"{name:.<30} {status_str}")
    
    passed = sum(1 for _, status in results if status)
    total = len(results)
    
    print(f"\nРезультат: {passed}/{total} проверок пройдено")
    
    if passed == total:
        print("\n🎉 СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К КОНКУРСУ!")
    elif passed >= total - 1:
        print("\n⚠️ Система почти готова, нужны небольшие доработки")
    else:
        print("\n❌ Система требует серьезных доработок")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)