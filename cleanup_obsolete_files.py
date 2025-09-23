#!/usr/bin/env python
"""
Скрипт для безопасного удаления устаревших файлов из src/submit
ВНИМАНИЕ: Этот скрипт только показывает, что будет удалено, но НЕ удаляет файлы!
"""
import os
from pathlib import Path

def analyze_obsolete_files():
    """Анализирует устаревшие файлы и показывает, что можно удалить"""
    
    submit_dir = Path("src/submit")
    
    # Файлы для удаления (безопасно)
    obsolete_files = [
        # Устаревшие версии model_inference
        "model_inference_original.py",
        "model_inference_v2.py", 
        "model_inference_vector.py",
        
        # Устаревшие компоненты RAG
        "rag/vector_rag_engine.py",
        "rag/vector_rag_interface.py",
        
        # Устаревшие компоненты embeddings
        "embeddings/embedding_engine.py",
        "embeddings/vector_store.py", 
        "embeddings/vector_models.py",
        "embeddings/vector_utils.py",
        "embeddings/test_vector_search.py",
        
        # Неиспользуемые компоненты
        "questions/classifier.py",
        "questions/confidence.py",
        "questions/topics.py",
        "ranking/scorer.py",
        "ranking/session_ranker.py",
        "prompts/fallback_prompts.py",
        "prompts/topic_prompts.py",
        
        # Устаревшие конфигурации
        "config_loader.py",
        "config.yaml",
        
        # Неиспользуемые компоненты core
        "core/data_loader.py",
        "core/message_filter.py",
    ]
    
    # Папки для удаления (если пустые)
    obsolete_dirs = [
        "questions",
        "ranking", 
        "prompts"
    ]
    
    print("🔍 АНАЛИЗ УСТАРЕВШИХ ФАЙЛОВ В src/submit")
    print("=" * 60)
    
    total_size = 0
    files_to_delete = []
    
    print("\n📋 ФАЙЛЫ ДЛЯ УДАЛЕНИЯ:")
    print("-" * 40)
    
    for file_path in obsolete_files:
        full_path = submit_dir / file_path
        
        if full_path.exists():
            size = full_path.stat().st_size
            total_size += size
            files_to_delete.append((file_path, size))
            print(f"✅ {file_path} ({size:,} байт)")
        else:
            print(f"❌ {file_path} (не найден)")
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"Файлов к удалению: {len(files_to_delete)}")
    print(f"Общий размер: {total_size:,} байт ({total_size/1024:.1f} KB)")
    
    print(f"\n📁 ПАПКИ ДЛЯ ПРОВЕРКИ:")
    print("-" * 40)
    
    for dir_name in obsolete_dirs:
        dir_path = submit_dir / dir_name
        if dir_path.exists():
            # Проверяем, есть ли файлы кроме __init__.py и __pycache__
            files_in_dir = []
            for item in dir_path.iterdir():
                if item.is_file() and not item.name.startswith('__'):
                    files_in_dir.append(item.name)
            
            if files_in_dir:
                print(f"⚠️  {dir_name}/ - содержит файлы: {files_in_dir}")
            else:
                print(f"✅ {dir_name}/ - можно удалить (только служебные файлы)")
        else:
            print(f"❌ {dir_name}/ - не найдена")
    
    print(f"\n✅ АКТИВНЫЕ ФАЙЛЫ (НЕ УДАЛЯТЬ):")
    print("-" * 40)
    
    active_files = [
        "model_inference_optimized.py",
        "smart_memory.py", 
        "smart_memory_optimized.py",
        "llm_inference.py",
        "storage/",
        "compression/",
        "extraction/",
        "filters/",
        "optimization/",
        "embeddings/improved_vector_search.py",
        "embeddings/improved_vector_store.py",
        "rag/compressed_rag_engine.py",
        "rag/config.py",
        "rag/engine.py", 
        "rag/interface.py"
    ]
    
    for file_path in active_files:
        full_path = submit_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path} - АКТИВЕН")
        else:
            print(f"❌ {file_path} - не найден")
    
    print(f"\n🚀 РЕКОМЕНДАЦИИ:")
    print("-" * 40)
    print("1. Создайте бэкап: cp -r src/submit src/submit_backup")
    print("2. Удалите файлы поэтапно для проверки")
    print("3. Протестируйте систему после каждого этапа")
    print("4. Обновите документацию")
    
    print(f"\n⚠️  ВНИМАНИЕ:")
    print("-" * 40)
    print("Этот скрипт только АНАЛИЗИРУЕТ файлы!")
    print("Для удаления используйте команды вручную или создайте отдельный скрипт.")
    print("Всегда создавайте бэкап перед удалением!")
    
    return files_to_delete, total_size

def generate_cleanup_commands():
    """Генерирует команды для удаления (для копирования)"""
    
    commands = [
        "# Команды для удаления устаревших файлов",
        "# ВНИМАНИЕ: Сначала создайте бэкап!",
        "",
        "# 1. Создание бэкапа",
        "cp -r src/submit src/submit_backup",
        "",
        "# 2. Удаление устаревших версий model_inference",
        "rm src/submit/model_inference_original.py",
        "rm src/submit/model_inference_v2.py", 
        "rm src/submit/model_inference_vector.py",
        "",
        "# 3. Удаление устаревших компонентов RAG",
        "rm src/submit/rag/vector_rag_engine.py",
        "rm src/submit/rag/vector_rag_interface.py",
        "",
        "# 4. Удаление устаревших компонентов embeddings",
        "rm src/submit/embeddings/embedding_engine.py",
        "rm src/submit/embeddings/vector_store.py",
        "rm src/submit/embeddings/vector_models.py", 
        "rm src/submit/embeddings/vector_utils.py",
        "rm src/submit/embeddings/test_vector_search.py",
        "",
        "# 5. Удаление неиспользуемых компонентов",
        "rm src/submit/questions/classifier.py",
        "rm src/submit/questions/confidence.py",
        "rm src/submit/questions/topics.py",
        "rm src/submit/ranking/scorer.py",
        "rm src/submit/ranking/session_ranker.py",
        "rm src/submit/prompts/fallback_prompts.py",
        "rm src/submit/prompts/topic_prompts.py",
        "",
        "# 6. Удаление устаревших конфигураций",
        "rm src/submit/config_loader.py",
        "rm src/submit/config.yaml",
        "",
        "# 7. Удаление неиспользуемых компонентов core",
        "rm src/submit/core/data_loader.py",
        "rm src/submit/core/message_filter.py",
        "",
        "# 8. Удаление пустых папок (если они пустые)",
        "rmdir src/submit/questions 2>/dev/null || true",
        "rmdir src/submit/ranking 2>/dev/null || true", 
        "rmdir src/submit/prompts 2>/dev/null || true",
        "",
        "# 9. Очистка кэша Python",
        "find src/submit -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        "find src/submit -name '*.pyc' -delete 2>/dev/null || true",
        "",
        "echo 'Очистка завершена! Проверьте систему.'"
    ]
    
    return commands

def main():
    """Основная функция"""
    print("🧹 АНАЛИЗ УСТАРЕВШИХ ФАЙЛОВ")
    print("=" * 60)
    
    # Анализируем файлы
    files_to_delete, total_size = analyze_obsolete_files()
    
    # Генерируем команды
    commands = generate_cleanup_commands()
    
    # Сохраняем команды в файл
    with open("cleanup_commands.sh", "w") as f:
        f.write("\n".join(commands))
    
    print(f"\n💾 Команды для удаления сохранены в: cleanup_commands.sh")
    print(f"📋 Для выполнения: chmod +x cleanup_commands.sh && ./cleanup_commands.sh")
    
    print(f"\n🎯 ИТОГ:")
    print(f"Можно безопасно удалить {len(files_to_delete)} файлов")
    print(f"Экономия места: {total_size:,} байт ({total_size/1024:.1f} KB)")
    print(f"Система станет чище и проще в поддержке!")

if __name__ == "__main__":
    main()
