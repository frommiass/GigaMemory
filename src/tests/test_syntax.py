"""
Тесты для проверки синтаксиса всех Python файлов в папке submit
"""
import ast
import sys
import os
import unittest
from pathlib import Path

# Добавляем путь к src в sys.path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Путь к папке submit
submit_dir = src_dir / "submit"


class TestSyntax(unittest.TestCase):
    """Тесты для проверки синтаксиса Python файлов"""
    
    def test_all_python_files_syntax(self):
        """Тест синтаксиса всех Python файлов в папке submit"""
        python_files = list(submit_dir.glob("*.py"))
        
        self.assertGreater(len(python_files), 0, "В папке submit не найдено Python файлов")
        
        for py_file in python_files:
            with self.subTest(file=py_file.name):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    # Проверяем синтаксис
                    ast.parse(source_code)
                    
                except SyntaxError as e:
                    self.fail(f"Синтаксическая ошибка в файле {py_file.name}: {e}")
                except Exception as e:
                    self.fail(f"Ошибка при чтении файла {py_file.name}: {e}")
    
    def test_import_statements_valid(self):
        """Тест валидности всех import statements"""
        python_files = list(submit_dir.glob("*.py"))
        
        for py_file in python_files:
            with self.subTest(file=py_file.name):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    # Парсим AST
                    tree = ast.parse(source_code)
                    
                    # Проверяем все import statements
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                # Проверяем, что имя модуля не пустое
                                self.assertTrue(alias.name, f"Пустое имя модуля в {py_file.name}")
                                
                        elif isinstance(node, ast.ImportFrom):
                            # Проверяем, что модуль указан
                            if node.module:
                                self.assertTrue(node.module, f"Пустое имя модуля в from import в {py_file.name}")
                            
                            # Проверяем, что есть импортируемые имена
                            self.assertTrue(node.names, f"Нет импортируемых имен в from import в {py_file.name}")
                            
                            for alias in node.names:
                                self.assertTrue(alias.name, f"Пустое имя в from import в {py_file.name}")
                    
                except SyntaxError as e:
                    self.fail(f"Синтаксическая ошибка в файле {py_file.name}: {e}")
                except Exception as e:
                    self.fail(f"Ошибка при анализе импортов в файле {py_file.name}: {e}")
    
    def test_function_definitions_valid(self):
        """Тест валидности определений функций"""
        python_files = list(submit_dir.glob("*.py"))
        
        for py_file in python_files:
            with self.subTest(file=py_file.name):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    tree = ast.parse(source_code)
                    
                    # Проверяем все определения функций
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Проверяем, что имя функции не пустое
                            self.assertTrue(node.name, f"Пустое имя функции в {py_file.name}")
                            
                            # Проверяем, что есть тело функции
                            self.assertTrue(node.body, f"Пустое тело функции {node.name} в {py_file.name}")
                    
                except SyntaxError as e:
                    self.fail(f"Синтаксическая ошибка в файле {py_file.name}: {e}")
                except Exception as e:
                    self.fail(f"Ошибка при анализе функций в файле {py_file.name}: {e}")
    
    def test_class_definitions_valid(self):
        """Тест валидности определений классов"""
        python_files = list(submit_dir.glob("*.py"))
        
        for py_file in python_files:
            with self.subTest(file=py_file.name):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    tree = ast.parse(source_code)
                    
                    # Проверяем все определения классов
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Проверяем, что имя класса не пустое
                            self.assertTrue(node.name, f"Пустое имя класса в {py_file.name}")
                            
                            # Проверяем, что есть тело класса
                            self.assertTrue(node.body, f"Пустое тело класса {node.name} в {py_file.name}")
                    
                except SyntaxError as e:
                    self.fail(f"Синтаксическая ошибка в файле {py_file.name}: {e}")
                except Exception as e:
                    self.fail(f"Ошибка при анализе классов в файле {py_file.name}: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
