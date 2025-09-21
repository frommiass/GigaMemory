"""
Тесты для проверки импортов модулей без внешних зависимостей
"""
import sys
import os
import unittest
from pathlib import Path

# Добавляем путь к src в sys.path для корректного импорта
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))


class TestBasicImports(unittest.TestCase):
    """Тесты для проверки импортов модулей без внешних зависимостей"""
    
    def test_prompts_imports(self):
        """Тест импортов модуля prompts (без внешних зависимостей)"""
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
            
            # Тестируем функции с параметрами
            memory_text = "Тестовая память"
            question = "Тестовый вопрос"
            result = get_memory_extraction_prompt(question, memory_text)
            self.assertIn(question, result)
            self.assertIn(memory_text, result)
            
        except ImportError as e:
            self.fail(f"Ошибка импорта prompts: {e}")
    
    def test_regex_patterns_imports(self):
        """Тест импортов модуля regex_patterns (без внешних зависимостей)"""
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
            
            # Проверяем, что регулярные выражения компилированы
            for pattern in COMPILED_PATTERNS:
                self.assertTrue(hasattr(pattern, 'search'))
                self.assertTrue(hasattr(pattern, 'match'))
            
            # Проверяем, что маркеры содержат ожидаемые значения
            self.assertIn('я', PERSONAL_MARKERS)
            self.assertIn('мой', PERSONAL_MARKERS)
            self.assertIn('мы', PERSONAL_MARKERS)
            
        except ImportError as e:
            self.fail(f"Ошибка импорта regex_patterns: {e}")
    
    def test_memory_storage_imports(self):
        """Тест импортов модуля memory_storage (без внешних зависимостей)"""
        try:
            from submit.memory_storage import MemoryStorage
            from models import Message
            
            # Проверяем, что класс можно создать
            storage = MemoryStorage()
            self.assertTrue(hasattr(storage, 'add_to_memory'))
            self.assertTrue(hasattr(storage, 'get_memory'))
            self.assertTrue(hasattr(storage, 'increment_session'))
            self.assertTrue(hasattr(storage, 'check_cache'))
            self.assertTrue(hasattr(storage, 'add_to_cache'))
            
            # Тестируем базовую функциональность
            dialogue_id = "test_dialogue"
            messages = [Message(role="user", content="test message")]
            
            storage.add_to_memory(dialogue_id, messages)
            retrieved_messages = storage.get_memory(dialogue_id)
            self.assertEqual(len(retrieved_messages), 1)
            self.assertEqual(retrieved_messages[0].content, "test message")
            
            session_id = storage.increment_session(dialogue_id)
            self.assertEqual(session_id, 1)
            
        except ImportError as e:
            self.fail(f"Ошибка импорта memory_storage: {e}")
    
    def test_message_filter_imports(self):
        """Тест импортов модуля message_filter (без внешних зависимостей)"""
        try:
            from submit.message_filter import (
                filter_user_message, 
                clean_user_messages, 
                process_sessions
            )
            from models import Message
            
            # Проверяем, что функции можно вызвать
            self.assertTrue(callable(filter_user_message))
            self.assertTrue(callable(clean_user_messages))
            self.assertTrue(callable(process_sessions))
            
            # Тестируем функцию фильтрации
            test_content = "Это тестовое сообщение с личной информацией"
            result = filter_user_message(test_content)
            self.assertIsInstance(result, bool)
            
            # Тестируем функцию очистки сообщений
            messages = [
                Message(role="user", content="Личное сообщение"),
                Message(role="assistant", content="Ответ ассистента"),
                Message(role="user", content="Еще одно личное сообщение")
            ]
            
            filtered = clean_user_messages(messages)
            self.assertIsInstance(filtered, list)
            
            # Тестируем функцию обработки сессий
            processed = process_sessions(messages, session_id=1)
            self.assertIsInstance(processed, list)
            
        except ImportError as e:
            self.fail(f"Ошибка импорта message_filter: {e}")
    
    def test_models_import(self):
        """Тест импорта модуля models"""
        try:
            from models import Message
            
            # Проверяем, что класс Message можно создать
            msg = Message(role="user", content="test")
            self.assertEqual(msg.role, "user")
            self.assertEqual(msg.content, "test")
            
            # Проверяем, что Message является dataclass
            self.assertTrue(hasattr(msg, '__dataclass_fields__'))
            
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
    unittest.main(verbosity=2)
