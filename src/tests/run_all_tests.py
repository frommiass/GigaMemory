"""
Скрипт для запуска всех тестов импортов и синтаксиса
"""
import unittest
import sys
import os
from pathlib import Path

# Добавляем путь к src в sys.path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))


def run_all_tests():
    """Запускает все тесты импортов и синтаксиса"""
    
    # Загружаем тесты
    loader = unittest.TestLoader()
    start_dir = str(current_dir)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Возвращаем код выхода
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
