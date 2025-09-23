"""
Загрузчик конфигурации системы
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Полная конфигурация системы"""
    
    # Пути
    model_path: str
    config_path: str
    
    # Компоненты
    embedding_config: Dict[str, Any]
    vector_store_config: Dict[str, Any]
    fact_extraction_config: Dict[str, Any]
    compression_config: Dict[str, Any]
    rag_config: Dict[str, Any]
    cache_config: Dict[str, Any]
    batch_config: Dict[str, Any]
    
    # Оптимизация
    optimization_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    persistence_config: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SystemConfig':
        """Загружает конфигурацию из YAML файла"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls(
            model_path=config['model']['path'],
            config_path=config_path,
            embedding_config=config.get('embedding', {}),
            vector_store_config=config.get('vector_store', {}),
            fact_extraction_config=config.get('fact_extraction', {}),
            compression_config=config.get('compression', {}),
            rag_config=config.get('rag', {}),
            cache_config=config.get('cache', {}),
            batch_config=config.get('batch', {}),
            optimization_config=config.get('optimization', {}),
            monitoring_config=config.get('monitoring', {}),
            persistence_config=config.get('persistence', {})
        )
    
    def to_smart_memory_config(self):
        """Преобразует в конфигурацию SmartMemory"""
        from .smart_memory import SmartMemoryConfig
        from .compression import CompressionLevel
        
        config = SmartMemoryConfig()
        
        # Векторный поиск
        config.use_vector_search = True
        config.embedding_model = self.embedding_config.get('model_name', 'cointegrated/rubert-tiny2')
        config.vector_top_k = self.vector_store_config.get('top_k', 5)
        
        # Извлечение фактов
        config.use_fact_extraction = self.fact_extraction_config.get('enabled', True)
        config.fact_min_confidence = self.fact_extraction_config.get('min_confidence', 0.5)
        
        # Сжатие
        config.use_compression = self.compression_config.get('enabled', True)
        level_str = self.compression_config.get('level', 'moderate')
        config.compression_level = CompressionLevel[level_str.upper()]
        
        # Интеграция
        config.use_hybrid_search = self.rag_config.get('use_hybrid_search', True)
        config.max_context_length = 2000
        
        return config


class ConfigManager:
    """Менеджер конфигурации"""
    
    _instance = None
    _config: Optional[SystemConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """Загружает конфигурацию"""
        if config_path is None:
            # Пытаемся найти конфиг в стандартных местах
            search_paths = [
                './config.yaml',
                './submit/config.yaml',
                '../config/config.yaml',
                os.environ.get('GIGAMEMORY_CONFIG', '')
            ]
            
            for path in search_paths:
                if path and Path(path).exists():
                    config_path = path
                    break
            
            if config_path is None:
                # Используем дефолтную конфигурацию
                config_path = Path(__file__).parent / 'config.yaml'
        
        logger.info(f"Загружаем конфигурацию из {config_path}")
        self._config = SystemConfig.from_yaml(config_path)
        return self._config
    
    def get_config(self) -> SystemConfig:
        """Возвращает текущую конфигурацию"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def reload_config(self):
        """Перезагружает конфигурацию"""
        if self._config:
            self._config = SystemConfig.from_yaml(self._config.config_path)
            logger.info("Конфигурация перезагружена")


# Глобальный экземпляр
config_manager = ConfigManager()
