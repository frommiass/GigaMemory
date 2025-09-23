# 🧹 Анализ устаревших файлов в корне проекта

## 📊 ТЕКУЩИЕ ФАЙЛЫ В КОРНЕ

### ✅ АКТИВНЫЕ ФАЙЛЫ (НЕ УДАЛЯТЬ)

#### Основные рабочие файлы
- `run.py` ✅ - главный скрипт запуска
- `requirements.txt` ✅ - зависимости
- `README.md` ✅ - основная документация
- `models.py` ✅ - модели данных
- `submit_interface.py` ✅ - интерфейс

#### Активные тесты
- `test_large_dialogue.py` ✅ - тест больших диалогов
- `test_optimized_with_facts.py` ✅ - тест с фактами
- `show_extracted_facts.py` ✅ - показ извлеченных фактов
- `validate_system.py` ✅ - валидация системы

#### Данные
- `data/format_example.jsonl` ✅ - данные лидерборда

### ❌ ФАЙЛЫ ДЛЯ УДАЛЕНИЯ

#### 1. Устаревшие тесты (20+ файлов)
**Старые тесты, замененные новыми:**

- `test_classification.py` ❌ - старый тест классификации
- `test_classification_simple.py` ❌ - упрощенный тест классификации
- `test_copypaste_filter.py` ❌ - старый тест фильтра копипаста
- `test_dataloader_filtering.py` ❌ - тест фильтрации данных
- `test_fact_extraction.py` ❌ - старый тест извлечения фактов
- `test_filtering_debug.py` ❌ - отладочный тест фильтрации
- `test_final_copypaste.py` ❌ - финальный тест копипаста
- `test_full_integration.py` ❌ - старый интеграционный тест
- `test_optimized_simple.py` ❌ - упрощенный тест оптимизированной системы
- `test_optimized_system.py` ❌ - старый тест оптимизированной системы
- `test_rag_components.py` ❌ - тест компонентов RAG
- `test_rag_integration.py` ❌ - интеграционный тест RAG
- `test_rag_mock.py` ❌ - мок-тест RAG
- `test_rag_simple.py` ❌ - простой тест RAG
- `test_real_copypaste.py` ❌ - реальный тест копипаста
- `test_session_numbering.py` ❌ - тест нумерации сессий
- `test_simple_system.py` ❌ - простой тест системы
- `test_simple_vector.py` ❌ - простой тест векторов
- `test_specific_message.py` ❌ - тест конкретного сообщения
- `test_system_mock.py` ❌ - мок-тест системы
- `test_vector_rag_integration.py` ❌ - интеграционный тест векторного RAG
- `test_vector_rag_isolated.py` ❌ - изолированный тест векторного RAG
- `test_vector_rag_simple.py` ❌ - простой тест векторного RAG
- `test_vector_search.py` ❌ - тест векторного поиска

**Причина:** Все эти тесты заменены на `test_large_dialogue.py` и `test_optimized_with_facts.py`

#### 2. Устаревшие демо-скрипты (3 файла)
- `demo_full_system.py` ❌ - демо полной системы
- `demo_improved_vector_search.py` ❌ - демо улучшенного векторного поиска
- `demo_prompt_generation.py` ❌ - демо генерации промптов

**Причина:** Функциональность интегрирована в основные компоненты

#### 3. Устаревшие скрипты извлечения (3 файла)
- `extract_clean_facts.py` ❌ - извлечение чистых фактов
- `extract_real_facts.py` ❌ - извлечение реальных фактов
- `extract_simple_facts.py` ❌ - извлечение простых фактов

**Причина:** Функциональность интегрирована в `smart_memory.py`

#### 4. Устаревшие MD файлы (8 файлов)
- `COPYPASTE_FIX_RESULTS.md` ❌ - результаты исправления копипаста
- `PROJECT_ANALYSIS.md` ❌ - анализ проекта
- `PROMPT_TEST_RESULTS.md` ❌ - результаты тестов промптов
- `RAG_ARCHITECTURE.md` ❌ - архитектура RAG
- `RAG_SYSTEM_TEST_RESULTS.md` ❌ - результаты тестов RAG
- `REFACTORING_PLAN.md` ❌ - план рефакторинга
- `REFACTORING_RESULTS.md` ❌ - результаты рефакторинга
- `obsolete_files_analysis.md` ❌ - анализ устаревших файлов

**Причина:** Устаревшая документация, заменена на `README.md` и `VECTOR_RAG_INTEGRATION.md`

#### 5. Устаревшие файлы интеграции (1 файл)
- `example_vector_rag_integration.py` ❌ - пример интеграции векторного RAG

**Причина:** Интеграция завершена, пример больше не нужен

#### 6. Временные файлы (3 файла)
- `simple_test_results.json` ❌ - результаты простых тестов
- `test_dialogue_100k.jsonl` ❌ - тестовый диалог 100k
- `extracted_facts_report.txt` ❌ - отчет об извлеченных фактах

**Причина:** Временные файлы, созданные для тестирования

#### 7. Устаревшие файлы в src/ (несколько файлов)
- `src/dialog_processor.py` ❌ - обработчик диалогов
- `src/models.py` ❌ - дублирует корневой models.py
- `src/run.py` ❌ - дублирует корневой run.py
- `src/test_classification.py` ❌ - дублирует корневой тест
- `src/tests/` ❌ - вся папка с устаревшими тестами
- `src/utils/` ❌ - утилиты, не используемые в активной системе
- `src/zip/` ❌ - архивные файлы

#### 8. Устаревшие файлы в start/ (вся папка)
- `start/` ❌ - вся папка с устаревшими скриптами запуска

**Причина:** Заменена на `run.py`

#### 9. Архивные файлы (2 файла)
- `giga_memory_stub.tar.gz` ❌ - архивный файл
- `giga_memory_stub.zip` ❌ - архивный файл

**Причина:** Архивные файлы, больше не нужны

#### 10. Служебные файлы (2 файла)
- `setup.cfg` ❌ - конфигурация setup
- `Dockerfile` ❌ - Docker файл (если не используется)

**Причина:** Не используются в текущей системе

## 📊 СТАТИСТИКА ОЧИСТКИ

**Файлов к удалению:** 50+ файлов
**Папок к удалению:** 4 папки (`src/tests/`, `src/utils/`, `src/zip/`, `start/`)
**Экономия места:** ~500+ KB

## 🚀 ПЛАН ОЧИСТКИ

### Этап 1: Удаление устаревших тестов
```bash
rm test_classification*.py
rm test_copypaste_filter.py
rm test_dataloader_filtering.py
rm test_fact_extraction.py
rm test_filtering_debug.py
rm test_final_copypaste.py
rm test_full_integration.py
rm test_optimized_simple.py
rm test_optimized_system.py
rm test_rag_*.py
rm test_real_copypaste.py
rm test_session_numbering.py
rm test_simple_*.py
rm test_specific_message.py
rm test_system_mock.py
rm test_vector_*.py
```

### Этап 2: Удаление демо и скриптов
```bash
rm demo_*.py
rm extract_*.py
rm example_vector_rag_integration.py
```

### Этап 3: Удаление устаревших MD файлов
```bash
rm COPYPASTE_FIX_RESULTS.md
rm PROJECT_ANALYSIS.md
rm PROMPT_TEST_RESULTS.md
rm RAG_ARCHITECTURE.md
rm RAG_SYSTEM_TEST_RESULTS.md
rm REFACTORING_PLAN.md
rm REFACTORING_RESULTS.md
rm obsolete_files_analysis.md
```

### Этап 4: Удаление временных файлов
```bash
rm simple_test_results.json
rm test_dialogue_100k.jsonl
rm extracted_facts_report.txt
```

### Этап 5: Удаление архивных файлов
```bash
rm giga_memory_stub.*
rm setup.cfg
rm Dockerfile
```

### Этап 6: Удаление устаревших папок
```bash
rm -rf src/tests/
rm -rf src/utils/
rm -rf src/zip/
rm -rf start/
```

### Этап 7: Удаление дублирующих файлов в src/
```bash
rm src/dialog_processor.py
rm src/models.py
rm src/run.py
rm src/test_classification.py
```

## ⚠️ ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ

1. **НЕ УДАЛЯТЬ:**
   - `run.py` - главный скрипт
   - `test_large_dialogue.py` - активный тест
   - `test_optimized_with_facts.py` - активный тест
   - `show_extracted_facts.py` - активный скрипт
   - `validate_system.py` - валидация системы
   - `README.md` - документация
   - `requirements.txt` - зависимости
   - `models.py` - модели данных
   - `data/` - данные

2. **Создать бэкап перед удалением:**
   ```bash
   cp -r . ../memory_aij2025_backup
   ```

3. **Тестировать после каждого этапа**

## 🎯 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

- **Упрощение структуры:** Удаление 50+ устаревших файлов
- **Улучшение читаемости:** Останутся только активные компоненты
- **Ускорение разработки:** Меньше файлов для анализа
- **Экономия места:** ~500+ KB

## ✅ ПРОВЕРКА ПОСЛЕ ОЧИСТКИ

```bash
# Проверка основных компонентов
python run.py --help
python test_large_dialogue.py
python show_extracted_facts.py
python validate_system.py
```
