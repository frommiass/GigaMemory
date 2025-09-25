# src/submit/bootstrap.py
"""
Bootstrap модуль - инициализация системы GigaMemory
ИСПРАВЛЕННАЯ ВЕРСИЯ с правильным порядком инициализации
"""

from .core.container import container
from .core.interfaces import *
from .core.orchestrator import MemoryOrchestrator
import yaml
import logging
import sys
import os

# Добавляем путь к models
sys.path.append(os.path.dirname(__file__) + '/../../')
from models import Message

logger = logging.getLogger(__name__)

def bootstrap_system(config_path: str = None):
    """Инициализирует систему с всеми модулями в ПРАВИЛЬНОМ ПОРЯДКЕ"""
    
    # Загружаем конфигурацию
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")
    
    logger.info("Starting system bootstrap...")
    
    try:
        # ============================================
        # КРИТИЧНО: СТРОГИЙ ПОРЯДОК ИНИЦИАЛИЗАЦИИ!
        # ============================================
        
        # 1. ПЕРВЫМ создаем OptimizationModule - он нужен всем
        logger.info("Initializing OptimizationModule...")
        from .modules.optimization.module import OptimizationModule
        optimizer = OptimizationModule(config.get('optimization', {}))
        container.register_singleton(IOptimizer, optimizer)
        logger.info("✅ OptimizationModule initialized")
        
        # 2. Storage - базовое хранилище
        logger.info("Initializing StorageModule...")
        from .modules.storage.module import StorageModule
        storage = StorageModule(config.get('storage', {}))
        storage.set_dependencies(optimizer=optimizer)
        container.register_singleton(IStorage, storage)
        logger.info("✅ StorageModule initialized")
        
        # 3. Embeddings - векторный поиск
        logger.info("Initializing EmbeddingsModule...")
        from .modules.embeddings.module import EmbeddingsModule
        embeddings = EmbeddingsModule(config.get('embeddings', {}))
        embeddings.set_dependencies(optimizer=optimizer, storage=storage)
        container.register_singleton(IEmbeddingEngine, embeddings)
        logger.info("✅ EmbeddingsModule initialized")
        
        # 4. Extraction - извлечение фактов
        logger.info("Initializing ExtractionModule...")
        from .modules.extraction.module import ExtractionModule
        extractor = ExtractionModule(config.get('extraction', {}))
        extractor.set_dependencies(
            optimizer=optimizer,
            storage=storage,
            embeddings=embeddings
        )
        container.register_singleton(IFactExtractor, extractor)
        logger.info("✅ ExtractionModule initialized")
        
        # 5. Compression - сжатие текста
        logger.info("Initializing CompressionModule...")
        from .modules.compression.module import CompressionModule
        compressor = CompressionModule(config.get('compression', {}))
        compressor.set_dependencies(optimizer=optimizer)
        container.register_singleton(ICompressor, compressor)
        logger.info("✅ CompressionModule initialized")
        
        # 6. RAG - генерация ответов (нужны ВСЕ предыдущие модули)
        logger.info("Initializing RAGModule...")
        from .modules.rag.module import RAGModule
        rag = RAGModule(config.get('rag', {}))
        rag.set_dependencies(
            storage=storage,
            embeddings=embeddings,
            extractor=extractor,
            compressor=compressor,
            optimizer=optimizer
        )
        container.register_singleton(IRAGEngine, rag)
        logger.info("✅ RAGModule initialized")
        
        # 7. ModelInference - последним, т.к. использует все остальное
        logger.info("Initializing ModelInference...")
        from .llm_inference import ModelInference
        model = ModelInference(config.get('model_path'))
        container.register_singleton(IModelInference, model)
        logger.info("✅ ModelInference initialized")
        
        # 8. Создаем оркестратор
        logger.info("Creating MemoryOrchestrator...")
        orchestrator = MemoryOrchestrator()
        logger.info("✅ MemoryOrchestrator created")
        
        # Проверка что все модули инициализированы
        _verify_initialization(orchestrator)
        
        logger.info("✅ System bootstrap completed successfully!")
        return orchestrator
        
    except Exception as e:
        logger.error(f"❌ Bootstrap failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to bootstrap system: {e}")

def bootstrap_system_with_config(config: dict):
    """
    Альтернативная функция для инициализации с готовым конфигом
    Используется в model_inference.py
    """
    logger.info("Bootstrap with inline config")
    
    try:
        # КРИТИЧНО: Тот же порядок что и выше!
        
        # 1. Optimizer ПЕРВЫМ
        from .modules.optimization.module import OptimizationModule
        optimizer = OptimizationModule(config.get('optimization', {}))
        container.register_singleton(IOptimizer, optimizer)
        
        # 2. Storage
        from .modules.storage.module import StorageModule  
        storage = StorageModule(config.get('storage', {}))
        storage.set_dependencies(optimizer=optimizer)
        container.register_singleton(IStorage, storage)
        
        # 3. Embeddings
        from .modules.embeddings.module import EmbeddingsModule
        embeddings = EmbeddingsModule(config.get('embeddings', {}))
        embeddings.set_dependencies(optimizer=optimizer, storage=storage)
        container.register_singleton(IEmbeddingEngine, embeddings)
        
        # 4. Extraction
        from .modules.extraction.module import ExtractionModule
        extractor = ExtractionModule(config.get('extraction', {}))
        extractor.set_dependencies(
            optimizer=optimizer,
            storage=storage,
            embeddings=embeddings
        )
        container.register_singleton(IFactExtractor, extractor)
        
        # 5. Compression
        from .modules.compression.module import CompressionModule
        compressor = CompressionModule(config.get('compression', {}))
        compressor.set_dependencies(optimizer=optimizer)
        container.register_singleton(ICompressor, compressor)
        
        # 6. RAG
        from .modules.rag.module import RAGModule
        rag = RAGModule(config.get('rag', {}))
        rag.set_dependencies(
            storage=storage,
            embeddings=embeddings,
            extractor=extractor,
            compressor=compressor,
            optimizer=optimizer
        )
        container.register_singleton(IRAGEngine, rag)
        
        # 7. Model
        from .llm_inference import ModelInference
        model = ModelInference(config.get('model_path'))
        container.register_singleton(IModelInference, model)
        
        # 8. Orchestrator
        orchestrator = MemoryOrchestrator()
        
        # Проверка
        _verify_initialization(orchestrator)
        
        logger.info("✅ Bootstrap with config completed")
        return orchestrator
        
    except Exception as e:
        logger.error(f"❌ Bootstrap with config failed: {e}")
        raise

def _verify_initialization(orchestrator):
    """Проверяет что все модули корректно инициализированы"""
    checks = [
        ('optimizer', orchestrator.optimizer),
        ('storage', orchestrator.storage),
        ('embeddings', orchestrator.embeddings),
        ('extractor', orchestrator.extractor),
        ('compressor', orchestrator.compressor),
        ('rag', orchestrator.rag),
        ('model', orchestrator.model)
    ]
    
    for name, module in checks:
        if module is None:
            raise RuntimeError(f"Module {name} is None after initialization!")
        logger.info(f"  ✓ {name} initialized: {type(module).__name__}")

def get_default_config():
    """Возвращает конфигурацию по умолчанию ДЛЯ КОНКУРСА"""
    return {
        'model_path': '/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16',
        
        'optimization': {
            'l1_cache_size': 200,        # Увеличили для скорости
            'l2_cache_size': 10000,
            'l2_cache_memory': 1024,
            'batch_size': 32,
            'num_workers': 4,
            'eviction_strategy': 'lru',
            'default_ttl': 3600,
            'disk_cache_path': '/tmp/gigamemory_cache'
        },
        
        'storage': {
            'cache_size': 10000,
            'filter_copypaste': True,
            'min_message_length': 10,
            'max_message_length': 10000
        },
        
        'embeddings': {
            'model_name': 'cointegrated/rubert-tiny2',
            'batch_size': 32,
            'device': 'cuda',
            'index_type': 'faiss',
            'metric': 'cosine',
            'use_faiss': True            # Включаем FAISS для скорости
        },
        
        'extraction': {
            'min_confidence': 0.6,
            'use_llm': False,            # Отключаем для скорости
            'use_rules': True,           # Используем правила
            'conflict_strategy': 'latest',
            'max_facts_per_session': 50,
            'filter_copypaste': True
        },
        
        'compression': {
            'level': 'moderate',
            'method': 'hybrid',
            'target_ratio': 0.3,
            'min_text_length': 500,
            'use_cache': True
        },
        
        'rag': {
            'top_k': 5,
            'use_hybrid_search': True,
            'use_compression': True,
            'max_context_length': 2000,
            'use_hierarchical': False
        }
    }

def warmup_system(orchestrator):
    """Прогревает систему для лучшей производительности"""
    logger.info("Warming up system...")
    
    try:
        # Прогреваем кэши
        test_messages = [
            Message("user", "Тестовое сообщение для прогрева"),
            Message("assistant", "Ответ для прогрева")
        ]
        
        # Тестовый прогон pipeline
        orchestrator.process_dialogue("warmup_test", test_messages)
        
        # Очищаем тестовые данные
        orchestrator.clear_dialogue("warmup_test")
        
        logger.info("✅ System warmed up")
        
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

# Экспортируем для использования
__all__ = [
    'bootstrap_system',
    'bootstrap_system_with_config',
    'get_default_config',
    'warmup_system'
]