# Команды для удаления устаревших файлов в корне
# ВНИМАНИЕ: Сначала создайте бэкап!

# 1. Создание бэкапа
cp -r . ../memory_aij2025_backup

# 2. Удаление устаревших тестов
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

# 3. Удаление демо и скриптов
rm demo_*.py
rm extract_*.py
rm example_vector_rag_integration.py

# 4. Удаление устаревших MD файлов
rm COPYPASTE_FIX_RESULTS.md
rm PROJECT_ANALYSIS.md
rm PROMPT_TEST_RESULTS.md
rm RAG_ARCHITECTURE.md
rm RAG_SYSTEM_TEST_RESULTS.md
rm REFACTORING_PLAN.md
rm REFACTORING_RESULTS.md
rm obsolete_files_analysis.md

# 5. Удаление временных файлов
rm simple_test_results.json
rm test_dialogue_100k.jsonl
rm extracted_facts_report.txt

# 6. Удаление архивных файлов
rm giga_memory_stub.*
rm setup.cfg
rm Dockerfile

# 7. Удаление дублирующих файлов в src/
rm src/dialog_processor.py
rm src/models.py
rm src/run.py
rm src/test_classification.py

# 8. Удаление устаревших папок
rm -rf src/tests/
rm -rf src/utils/
rm -rf src/zip/
rm -rf start/

# 9. Очистка кэша Python
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

echo 'Очистка корня завершена! Проверьте систему.'