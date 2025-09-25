#!/usr/bin/env python3
"""
Quick fix script - применяет все критические исправления
Запуск: python src/submit/quick_fix.py
"""

import os
import sys
import shutil
from pathlib import Path

def backup_file(filepath):
    """Создает резервную копию файла"""
    backup_path = f"{filepath}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backup created: {backup_path}")

def apply_fix_embeddings():
    """Исправляет EmbeddingsModule"""
    print("\n🔧 Fixing EmbeddingsModule...")
    
    filepath = "src/submit/modules/embeddings/module.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Заменяем set_optimizer на set_dependencies
        if "def set_optimizer(" in content:
            content = content.replace(
                "def set_optimizer(self, optimizer):",
                "def set_dependencies(self, optimizer=None, storage=None, embeddings=None):"
            )
            print("  ✅ Replaced set_optimizer -> set_dependencies")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ EmbeddingsModule fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing EmbeddingsModule: {e}")
        return False

def apply_fix_compression():
    """Исправляет CompressionModule"""
    print("\n🔧 Fixing CompressionModule...")
    
    filepath = "src/submit/modules/compression/module.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Заменяем set_optimizer на set_dependencies
        if "def set_optimizer(" in content:
            content = content.replace(
                "def set_optimizer(self, optimizer):",
                "def set_dependencies(self, optimizer=None):"
            )
            print("  ✅ Replaced set_optimizer -> set_dependencies")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ CompressionModule fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing CompressionModule: {e}")
        return False

def apply_fix_llm_inference():
    """Исправляет ModelInference"""
    print("\n🔧 Fixing ModelInference...")
    
    filepath = "src/submit/llm_inference.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Добавляем импорты если их нет
        if "from core.interfaces import IModelInference" not in content:
            import_line = "from core.interfaces import IModelInference, ProcessingResult\n"
            content = import_line + content
            print("  ✅ Added interface imports")
        
        # Добавляем наследование от интерфейса
        if "class ModelInference:" in content and "IModelInference" not in content:
            content = content.replace(
                "class ModelInference:",
                "class ModelInference(IModelInference):"
            )
            print("  ✅ Added interface inheritance")
        
        # Добавляем метод generate если его нет
        if "def generate(self, messages: List[Message]) -> ProcessingResult:" not in content:
            generate_method = '''
    def generate(self, messages: List[Message]) -> ProcessingResult:
        """Реализация метода интерфейса IModelInference"""
        try:
            result = self.inference(messages)
            return ProcessingResult(
                success=True,
                data=result,
                metadata={
                    'model_path': self.model_path,
                    'messages_count': len(messages)
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data="Ошибка генерации.",
                metadata={'error': str(e)},
                error=str(e)
            )
'''
            # Вставляем после __init__
            insert_pos = content.find("def inference(")
            if insert_pos > 0:
                content = content[:insert_pos] + generate_method + "\n" + content[insert_pos:]
                print("  ✅ Added generate method")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ ModelInference fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing ModelInference: {e}")
        return False

def apply_fix_orchestrator():
    """Исправляет Orchestrator"""
    print("\n🔧 Fixing Orchestrator...")
    
    filepath = "src/submit/core/orchestrator.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Добавляем RAG в _setup_dependencies если его нет
        if "self.rag.set_dependencies" not in content:
            rag_setup = '''
    # RAG использует все модули
    if hasattr(self.rag, 'set_dependencies'):
        self.rag.set_dependencies(
            storage=self.storage,
            embeddings=self.embeddings,
            extractor=self.extractor,
            compressor=self.compressor,
            optimizer=self.optimizer
        )'''
            
            # Вставляем перед logger.info
            insert_pos = content.find('logger.info("Module dependencies configured")')
            if insert_pos > 0:
                content = content[:insert_pos] + rag_setup + "\n\n    " + content[insert_pos:]
                print("  ✅ Added RAG dependencies setup")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Orchestrator fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing Orchestrator: {e}")
        return False

def main():
    print("="*50)
    print("🚀 GIGAMEMORY QUICK FIX")
    print("="*50)
    
    # Применяем фиксы
    results = []
    
    results.append(("EmbeddingsModule", apply_fix_embeddings()))
    results.append(("CompressionModule", apply_fix_compression()))
    results.append(("ModelInference", apply_fix_llm_inference()))
    results.append(("Orchestrator", apply_fix_orchestrator()))
    
    # Отчет
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ:")
    print("="*50)
    
    for module, success in results:
        status = "✅ FIXED" if success else "❌ FAILED"
        print(f"{module:.<30} {status}")
    
    if all(success for _, success in results):
        print("\n✅ Все фиксы применены успешно!")
        print("Теперь запустите: python src/submit/final_integration.py")
    else:
        print("\n⚠️ Некоторые фиксы не удалось применить")

if __name__ == "__main__":
    main()
