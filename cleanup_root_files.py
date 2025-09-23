#!/usr/bin/env python
"""
Скрипт для безопасного удаления устаревших файлов в корне проекта
ВНИМАНИЕ: Этот скрипт только показывает, что будет удалено, но НЕ удаляет файлы!
"""
import os
from pathlib import Path

def analyze_root_files():
    """Анализирует файлы в корне и показывает, что можно удалить"""
    
    # Файлы для удаления (безопасно)
    obsolete_files = [
        # Устаревшие тесты
        "test_classification.py",
        "test_classification_simple.py", 
        "test_copypaste_filter.py",
        "test_dataloader_filtering.py",
        "test_fact_extraction.py",
        "test_filtering_debug.py",
        "test_final_copypaste.py",
        "test_full_integration.py",
        "test_optimized_simple.py",
        "test_optimized_system.py",
        "test_rag_components.py",
        "test_rag_integration.py",
        "test_rag_mock.py",
        "test_rag_simple.py",
        "test_real_copypaste.py",
        "test_session_numbering.py",
        "test_simple_system.py",
        "test_simple_vector.py",
        "test_specific_message.py",
        "test_system_mock.py",
        "test_vector_rag_integration.py",
        "test_vector_rag_isolated.py",
        "test_vector_rag_simple.py",
        "test_vector_search.py",
        
        # Демо скрипты
        "demo_full_system.py",
        "demo_improved_vector_search.py",
        "demo_prompt_generation.py",
        
        # Скрипты извлечения
        "extract_clean_facts.py",
        "extract_real_facts.py",
        "extract_simple_facts.py",
        
        # Устаревшие MD файлы
        "COPYPASTE_FIX_RESULTS.md",
        "PROJECT_ANALYSIS.md",
        "PROMPT_TEST_RESULTS.md",
        "RAG_ARCHITECTURE.md",
        "RAG_SYSTEM_TEST_RESULTS.md",
        "REFACTORING_PLAN.md",
        "REFACTORING_RESULTS.md",
        "obsolete_files_analysis.md",
        
        # Файлы интеграции
        "example_vector_rag_integration.py",
        
        # Временные файлы
        "simple_test_results.json",
        "test_dialogue_100k.jsonl",
        "extracted_facts_report.txt",
        
        # Архивные файлы
        "giga_memory_stub.tar.gz",
        "giga_memory_stub.zip",
        
        # Служебные файлы
        "setup.cfg",
        "Dockerfile",
        
        # Дублирующие файлы в src/
        "src/dialog_processor.py",
        "src/models.py",
        "src/run.py",
        "src/test_classification.py",
    ]
    
    # Папки для удаления
    obsolete_dirs = [
        "src/tests",
        "src/utils", 
        "src/zip",
        "start"
    ]
    
    print("🔍 АНАЛИЗ УСТАРЕВШИХ ФАЙЛОВ В КОРНЕ ПРОЕКТА")
    print("=" * 60)
    
    total_size = 0
    files_to_delete = []
    
    print("\n📋 ФАЙЛЫ ДЛЯ УДАЛЕНИЯ:")
    print("-" * 40)
    
    for file_path in obsolete_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            files_to_delete.append((file_path, size))
            print(f"✅ {file_path} ({size:,} байт)")
        else:
            print(f"❌ {file_path} (не найден)")
    
    print(f"\n📁 ПАПКИ ДЛЯ УДАЛЕНИЯ:")
    print("-" * 40)
    
    for dir_name in obsolete_dirs:
        if os.path.exists(dir_name):
            # Подсчитываем размер папки
            dir_size = 0
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        dir_size += os.path.getsize(file_path)
                    except:
                        pass
            
            total_size += dir_size
            print(f"✅ {dir_name}/ ({dir_size:,} байт)")
        else:
            print(f"❌ {dir_name}/ (не найдена)")
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"Файлов к удалению: {len(files_to_delete)}")
    print(f"Папок к удалению: {len([d for d in obsolete_dirs if os.path.exists(d)])}")
    print(f"Общий размер: {total_size:,} байт ({total_size/1024:.1f} KB)")
    
    print(f"\n✅ АКТИВНЫЕ ФАЙЛЫ (НЕ УДАЛЯТЬ):")
    print("-" * 40)
    
    active_files = [
        "run.py",
        "requirements.txt", 
        "README.md",
        "models.py",
        "submit_interface.py",
        "test_large_dialogue.py",
        "test_optimized_with_facts.py",
        "show_extracted_facts.py",
        "validate_system.py",
        "data/",
        "src/submit/",
        "VECTOR_RAG_INTEGRATION.md",
        "SYSTEM_READINESS_REPORT.md",
        "CLEANUP_REPORT.md"
    ]
    
    for file_path in active_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - АКТИВЕН")
        else:
            print(f"❌ {file_path} - не найден")
    
    print(f"\n🚀 РЕКОМЕНДАЦИИ:")
    print("-" * 40)
    print("1. Создайте бэкап: cp -r . ../memory_aij2025_backup")
    print("2. Удалите файлы поэтапно для проверки")
    print("3. Протестируйте систему после каждого этапа")
    print("4. Обновите документацию")
    
    print(f"\n⚠️  ВНИМАНИЕ:")
    print("-" * 40)
    print("Этот скрипт только АНАЛИЗИРУЕТ файлы!")
    print("Для удаления используйте команды вручную или создайте отдельный скрипт.")
    print("Всегда создавайте бэкап перед удалением!")
    
    return files_to_delete, total_size

def generate_root_cleanup_commands():
    """Генерирует команды для удаления файлов в корне"""
    
    commands = [
        "# Команды для удаления устаревших файлов в корне",
        "# ВНИМАНИЕ: Сначала создайте бэкап!",
        "",
        "# 1. Создание бэкапа",
        "cp -r . ../memory_aij2025_backup",
        "",
        "# 2. Удаление устаревших тестов",
        "rm test_classification*.py",
        "rm test_copypaste_filter.py",
        "rm test_dataloader_filtering.py", 
        "rm test_fact_extraction.py",
        "rm test_filtering_debug.py",
        "rm test_final_copypaste.py",
        "rm test_full_integration.py",
        "rm test_optimized_simple.py",
        "rm test_optimized_system.py",
        "rm test_rag_*.py",
        "rm test_real_copypaste.py",
        "rm test_session_numbering.py",
        "rm test_simple_*.py",
        "rm test_specific_message.py",
        "rm test_system_mock.py",
        "rm test_vector_*.py",
        "",
        "# 3. Удаление демо и скриптов",
        "rm demo_*.py",
        "rm extract_*.py",
        "rm example_vector_rag_integration.py",
        "",
        "# 4. Удаление устаревших MD файлов",
        "rm COPYPASTE_FIX_RESULTS.md",
        "rm PROJECT_ANALYSIS.md",
        "rm PROMPT_TEST_RESULTS.md",
        "rm RAG_ARCHITECTURE.md",
        "rm RAG_SYSTEM_TEST_RESULTS.md",
        "rm REFACTORING_PLAN.md",
        "rm REFACTORING_RESULTS.md",
        "rm obsolete_files_analysis.md",
        "",
        "# 5. Удаление временных файлов",
        "rm simple_test_results.json",
        "rm test_dialogue_100k.jsonl",
        "rm extracted_facts_report.txt",
        "",
        "# 6. Удаление архивных файлов",
        "rm giga_memory_stub.*",
        "rm setup.cfg",
        "rm Dockerfile",
        "",
        "# 7. Удаление дублирующих файлов в src/",
        "rm src/dialog_processor.py",
        "rm src/models.py",
        "rm src/run.py",
        "rm src/test_classification.py",
        "",
        "# 8. Удаление устаревших папок",
        "rm -rf src/tests/",
        "rm -rf src/utils/",
        "rm -rf src/zip/",
        "rm -rf start/",
        "",
        "# 9. Очистка кэша Python",
        "find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        "find . -name '*.pyc' -delete 2>/dev/null || true",
        "",
        "echo 'Очистка корня завершена! Проверьте систему.'"
    ]
    
    return commands

def main():
    """Основная функция"""
    print("🧹 АНАЛИЗ УСТАРЕВШИХ ФАЙЛОВ В КОРНЕ")
    print("=" * 60)
    
    # Анализируем файлы
    files_to_delete, total_size = analyze_root_files()
    
    # Генерируем команды
    commands = generate_root_cleanup_commands()
    
    # Сохраняем команды в файл
    with open("cleanup_root_commands.sh", "w") as f:
        f.write("\n".join(commands))
    
    print(f"\n💾 Команды для удаления сохранены в: cleanup_root_commands.sh")
    print(f"📋 Для выполнения: chmod +x cleanup_root_commands.sh && ./cleanup_root_commands.sh")
    
    print(f"\n🎯 ИТОГ:")
    print(f"Можно безопасно удалить {len(files_to_delete)} файлов и несколько папок")
    print(f"Экономия места: {total_size:,} байт ({total_size/1024:.1f} KB)")
    print(f"Проект станет значительно чище!")

if __name__ == "__main__":
    main()
