"""
Тесты для проверки корректности импортов всех модулей в папке submit
"""
import sys
import os
import importlib
import unittest
from pathlib import Path

# Добавляем путь к src в sys.path для корректного импорта
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Путь к папке submit
submit_dir = src_dir / "submit"


class TestImports(unittest.TestCase):
    """Тесты для проверки импортов модулей submit"""
    
    def test_llm_inference_imports(self):
        """Тест импортов модуля llm_inference"""
        try:
            from submit.llm_inference import ModelInference
            # Проверяем, что класс можно создать (без реальной инициализации модели)
            self.assertTrue(hasattr(ModelInference, '__init__'))
            self.assertTrue(hasattr(ModelInference, 'inference'))
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Модуль llm_inference требует внешние зависимости: {e}")
            else:
                self.fail(f"Ошибка импорта llm_inference: {e}")
        except Exception as e:
            # Ожидаем ошибку при инициализации без модели
            self.assertIn("Ошибка загрузки модели", str(e))
    
    def test_memory_storage_imports(self):
        """Тест импортов модуля memory_storage"""
        try:
            from submit.memory_storage import MemoryStorage
            # Проверяем, что класс можно создать
            storage = MemoryStorage()
            self.assertTrue(hasattr(storage, 'add_to_memory'))
            self.assertTrue(hasattr(storage, 'get_memory'))
            self.assertTrue(hasattr(storage, 'increment_session'))
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Модуль memory_storage требует внешние зависимости: {e}")
            else:
                self.fail(f"Ошибка импорта memory_storage: {e}")
    
    def test_message_filter_imports(self):
        """Тест импортов модуля message_filter"""
        try:
            from submit.message_filter import (
                filter_user_message, 
                clean_user_messages, 
                process_sessions
            )
            # Проверяем, что функции можно вызвать
            self.assertTrue(callable(filter_user_message))
            self.assertTrue(callable(clean_user_messages))
            self.assertTrue(callable(process_sessions))
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Модуль message_filter требует внешние зависимости: {e}")
            else:
                self.fail(f"Ошибка импорта message_filter: {e}")
    
    def test_model_inference_imports(self):
        """Тест импортов модуля model_inference"""
        try:
            from submit.model_inference import SubmitModelWithMemory
            # Проверяем, что класс можно создать (без реальной инициализации модели)
            self.assertTrue(hasattr(SubmitModelWithMemory, '__init__'))
            self.assertTrue(hasattr(SubmitModelWithMemory, 'write_to_memory'))
            self.assertTrue(hasattr(SubmitModelWithMemory, 'clear_memory'))
            self.assertTrue(hasattr(SubmitModelWithMemory, 'answer_to_question'))
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Модуль model_inference требует внешние зависимости: {e}")
            else:
                self.fail(f"Ошибка импорта model_inference: {e}")
        except Exception as e:
            # Ожидаем ошибку при инициализации без модели
            self.assertIn("Ошибка загрузки модели", str(e))
    
    def test_prompts_imports(self):
        """Тест импортов модуля prompts"""
        try:
            from submit.prompts import (
                get_memory_extraction_prompt,
                get_session_marker_prompt,
                get_personal_info_marker,
                get_memory_extraction_prompt_v2,
                get_memory_extraction_prompt_v3
            )
            # Проверяем, что функции можно вызвать
            self.assertTrue(callable(get_memory_extraction_prompt))
            self.assertTrue(callable(get_session_marker_prompt))
            self.assertTrue(callable(get_personal_info_marker))
            self.assertTrue(callable(get_memory_extraction_prompt_v2))
            self.assertTrue(callable(get_memory_extraction_prompt_v3))
            
            # Тестируем вызовы функций
            result = get_session_marker_prompt(1)
            self.assertIn("Сессия 1", result)
            
            result = get_personal_info_marker()
            self.assertEqual(result, "[ЛИЧНОЕ]")
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Модуль prompts требует внешние зависимости: {e}")
            else:
                self.fail(f"Ошибка импорта prompts: {e}")
    
    def test_regex_patterns_imports(self):
        """Тест импортов модуля regex_patterns"""
        try:
            from submit.regex_patterns import (
                COMPILED_PATTERNS,
                PERSONAL_MARKERS,
                COPYPASTE_MARKERS,
                TECH_SIGNS
            )
            # Проверяем типы данных
            self.assertIsInstance(COMPILED_PATTERNS, list)
            self.assertIsInstance(PERSONAL_MARKERS, frozenset)
            self.assertIsInstance(COPYPASTE_MARKERS, frozenset)
            self.assertIsInstance(TECH_SIGNS, frozenset)
            
            # Проверяем, что есть элементы
            self.assertGreater(len(COMPILED_PATTERNS), 0)
            self.assertGreater(len(PERSONAL_MARKERS), 0)
            self.assertGreater(len(COPYPASTE_MARKERS), 0)
            self.assertGreater(len(TECH_SIGNS), 0)
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Модуль regex_patterns требует внешние зависимости: {e}")
            else:
                self.fail(f"Ошибка импорта regex_patterns: {e}")
    
    def test_all_modules_importable(self):
        """Тест, что все модули в папке submit можно импортировать"""
        submit_files = [
            'llm_inference',
            'memory_storage', 
            'message_filter',
            'model_inference',
            'prompts',
            'regex_patterns'
        ]
        
        for module_name in submit_files:
            with self.subTest(module=module_name):
                try:
                    # Импортируем модуль напрямую, минуя __init__.py
                    module = importlib.import_module(f'submit.{module_name}')
                    self.assertIsNotNone(module)
                except ImportError as e:
                    # Если ошибка связана с отсутствующими внешними зависимостями,
                    # это нормально для тестов импортов
                    if any(dep in str(e) for dep in ['vllm', 'transformers']):
                        self.skipTest(f"Модуль {module_name} требует внешние зависимости: {e}")
                    else:
                        self.fail(f"Не удалось импортировать модуль {module_name}: {e}")
    
    def test_cross_module_dependencies(self):
        """Тест зависимостей между модулями"""
        try:
            # Проверяем, что модули могут импортировать друг друга
            from submit.message_filter import filter_user_message
            from submit.regex_patterns import COMPILED_PATTERNS
            from submit.prompts import get_session_marker_prompt
            from submit.memory_storage import MemoryStorage
            from submit.model_inference import SubmitModelWithMemory
            
            # Проверяем, что функции работают с импортированными данными
            test_content = "Это тестовое сообщение"
            result = filter_user_message(test_content)
            self.assertIsInstance(result, bool)
            
        except ImportError as e:
            if any(dep in str(e) for dep in ['vllm', 'transformers']):
                self.skipTest(f"Межмодульные зависимости требуют внешние библиотеки: {e}")
            else:
                self.fail(f"Ошибка в межмодульных зависимостях: {e}")
    
    def test_models_import(self):
        """Тест импорта модуля models"""
        try:
            from models import Message
            # Проверяем, что класс Message можно создать
            msg = Message(role="user", content="test")
            self.assertEqual(msg.role, "user")
            self.assertEqual(msg.content, "test")
        except ImportError as e:
            self.fail(f"Ошибка импорта models: {e}")
    
    def test_submit_interface_import(self):
        """Тест импорта модуля submit_interface"""
        try:
            from submit_interface import ModelWithMemory
            # Проверяем, что класс можно импортировать
            self.assertTrue(hasattr(ModelWithMemory, '__init__'))
        except ImportError as e:
            self.fail(f"Ошибка импорта submit_interface: {e}")


if __name__ == '__main__':
    # Запуск тестов
    unittest.main(verbosity=2)
