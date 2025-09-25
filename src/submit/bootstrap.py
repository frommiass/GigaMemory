# src/submit/bootstrap.py

from .core.container import container
from .core.interfaces import *
from .core.orchestrator import MemoryOrchestrator
from .modules.storage.module import StorageModule
from .modules.embeddings.module import EmbeddingsModule
from .modules.extraction.module import ExtractionModule
from .modules.compression.module import CompressionModule
from .modules.rag.module import RAGModule
from .llm_inference import ModelInference
import yaml

def bootstrap_system(config_path: str = None):
    """Инициализирует систему"""
    
    # Загружаем конфигурацию
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # Регистрируем модули
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
    
    container.register_singleton(
        IModelInference,
        ModelInference(config.get('model_path'))
    )
    
    # Создаем оркестратор
    orchestrator = MemoryOrchestrator()
    
    return orchestrator

def get_default_config():
    """Возвращает конфигурацию по умолчанию"""
    return {
        'model_path': '/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16',
        'storage': {
            'cache_size': 10000,
            'filter_copypaste': True
        },
        'embeddings': {
            'model_name': 'cointegrated/rubert-tiny2',
            'batch_size': 32,
            'device': 'cuda'
        },
        'extraction': {
            'min_confidence': 0.6,
            'use_llm': True
        },
        'compression': {
            'level': 'moderate',
            'method': 'hybrid',
            'target_ratio': 0.3
        },
        'rag': {
            'top_k': 5,
            'use_hybrid_search': True
        }
    }