# Тесты для модулей submit

Этот пакет содержит тесты для проверки корректности импортов и синтаксиса всех модулей в папке `submit`.

## Структура тестов

### `test_basic_imports.py`
Тесты для модулей без внешних зависимостей:
- `test_memory_storage_imports` - тест импортов модуля `memory_storage`
- `test_message_filter_imports` - тест импортов модуля `message_filter`
- `test_prompts_imports` - тест импортов модуля `prompts`
- `test_regex_patterns_imports` - тест импортов модуля `regex_patterns`
- `test_models_import` - тест импорта модуля `models`
- `test_submit_interface_import` - тест импорта модуля `submit_interface`

### `test_imports.py`
Расширенные тесты импортов, включая модули с внешними зависимостями:
- Тесты для всех модулей в папке `submit`
- Тесты межмодульных зависимостей
- Автоматический пропуск тестов для модулей, требующих внешние библиотеки (vllm, transformers)

### `test_syntax.py`
Тесты синтаксиса Python файлов:
- `test_all_python_files_syntax` - проверка синтаксиса всех Python файлов
- `test_import_statements_valid` - проверка валидности import statements
- `test_function_definitions_valid` - проверка валидности определений функций
- `test_class_definitions_valid` - проверка валидности определений классов

## Запуск тестов

### Запуск всех тестов:
```bash
python src/tests/run_all_tests.py
```

### Запуск отдельных групп тестов:
```bash
# Только базовые тесты импортов
python src/tests/test_basic_imports.py

# Только тесты синтаксиса
python src/tests/test_syntax.py

# Все тесты импортов (включая с внешними зависимостями)
python src/tests/test_imports.py
```

## Результаты

Все тесты проходят успешно:
- ✅ 6 тестов базовых импортов
- ✅ 4 теста синтаксиса
- ⏭️ 5 тестов пропущены (требуют внешние зависимости: vllm, transformers)
- ✅ 5 тестов расширенных импортов

**Итого: 15 тестов прошли, 5 пропущены**

## Примечания

- Тесты автоматически пропускают модули, требующие внешние зависимости (vllm, transformers)
- Все модули без внешних зависимостей работают корректно
- Синтаксис всех Python файлов корректен
- Импорты между модулями настроены правильно
