"""
Пакет для семантического сжатия информации
"""

from .compression_models import (
    CompressionLevel,
    CompressionMethod,
    CompressionResult,
    CompressionStats,
    CompressionConfig
)

from .semantic_compressor import (
    SemanticCompressor,
    RuleBasedCompressor,
    LLMCompressor,
    HybridCompressor
)

from .hierarchical_compressor import (
    HierarchicalCompressor,
    HierarchyLevel,
    CompressionHierarchy
)

from .compression_strategies import (
    CompressionStrategy,
    AbstractiveStrategy,
    ExtractiveStrategy,
    TemplateStrategy,
    get_strategy
)

__all__ = [
    # Models
    'CompressionLevel',
    'CompressionMethod',
    'CompressionResult',
    'CompressionStats',
    'CompressionConfig',
    
    # Compressors
    'SemanticCompressor',
    'RuleBasedCompressor',
    'LLMCompressor',
    'HybridCompressor',
    
    # Hierarchical
    'HierarchicalCompressor',
    'HierarchyLevel',
    'CompressionHierarchy',
    
    # Strategies
    'CompressionStrategy',
    'AbstractiveStrategy',
    'ExtractiveStrategy',
    'TemplateStrategy',
    'get_strategy'
]

