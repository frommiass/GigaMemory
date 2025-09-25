# src/submit/bootstrap.py

from .core.container import container
from .core.interfaces import *
from .core.orchestrator import MemoryOrchestrator
from .modules.storage.module import StorageModule
from .modules.embeddings.module import EmbeddingsModule
from .modules.extraction.module import ExtractionModule
from .modules.compression.module import CompressionModule
from .modules.optimization.module import OptimizationModule
from .modules.rag.module import RAGModule
from .llm_inference import ModelInference
import yaml
import logging

logger = logging.getLogger(__name__)

def bootstrap_system(config_path: str = None):
    """Инициализирует систему с всеми модулями"""
    
    # Загружаем конфигурацию
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    logger.info("Bootstrapping system with configuration")
    
    # ВАЖНО: Регистрируем OptimizationModule ПЕРВЫМ
    container.register_factory(
        IOptimizer,
        lambda: OptimizationModule(config.get('optimization', {}))
    )
    
    # Регистрируем остальные модули
    container.register_factory(
        IStorage,
        lambda: StorageModule(config.get('storage', {}))
    )
    
    container.register_factory(
        IEmbeddingEngine,
        lambda: EmbeddingsModule(config.get('embeddings', {}))
    )
    
    container.register_factory(
        IFactExtractor,
        lambda: ExtractionModule(config.get('extraction', {}))
    )
    
    container.register_factory(
        ICompressor,
        lambda: CompressionModule(config.get('compression', {}))
    )
    
    container.register_factory(
        IRAGEngine,
        lambda: RAGModule(config.get('rag', {}))
    )
    
    # ModelInference как singleton
    container.register_singleton(
        IModelInference,
        ModelInference(config.get('model_path'))
    )
    
    # Создаем оркестратор (он сам получит все модули и настроит зависимости)
    orchestrator = MemoryOrchestrator()
    
    logger.info("System bootstrap complete")
    return orchestrator

def get_default_config():
    """Возвращает конфигурацию по умолчанию"""
    return {
        'model_path': '/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16',
        'optimization': {
            'l1_cache_size': 100,
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
            'metric': 'cosine'
        },
        'extraction': {
            'min_confidence': 0.6,
            'use_llm': False,  # Для скорости используем rules
            'use_rules': True,
            'conflict_strategy': 'latest',
            'max_facts_per_session': 50
        },
        'compression': {
            'level': 'moderate',
            'method': 'hybrid',
            'target_ratio': 0.3,
            'min_text_length': 500
        },
        'rag': {
            'top_k': 5,
            'use_hybrid_search': True,
            'use_compression': True,
            'max_context_length': 2000
        }
    }