"""
Модуль извлечения фактов для GigaMemory
Версия 2.0 с улучшенными паттернами и извлечением
"""

from .fact_models import (
    Fact,
    FactType,
    FactRelation,
    FactConfidence,
    TemporalFact
)

from .fact_extractor import (
    FactExtractor,
    RuleBasedFactExtractor,
    SmartFactExtractor,
    ExtractionStats
)

from .fact_database import (
    FactDatabase,
    FactStats,
    FactIndex,
    FactConflictResolver
)

from .fact_patterns import (
    FACT_PATTERNS,
    extract_with_pattern,
    extract_all_with_patterns,
    detect_temporal_context,
    normalize_value,
    confidence_from_pattern_match,
    get_relation_for_type
)

from .patterns import (
    FactPattern,
    FactPatterns
)

from .module import ExtractionModule

__version__ = "2.0.0"
__author__ = "GigaMemory Team"

__all__ = [
    # Основные классы
    "FactExtractor",
    "RuleBasedFactExtractor", 
    "SmartFactExtractor",
    "FactDatabase",
    "ExtractionModule",
    "FactPatterns",
    
    # Модели данных
    "Fact",
    "TemporalFact",
    "FactStats",
    "FactIndex",
    "FactConflictResolver",
    "ExtractionStats",
    "FactPattern",
    
    # Перечисления
    "FactType",
    "FactRelation",
    "FactConfidence",
    
    # Утилиты
    "FACT_PATTERNS",
    "extract_with_pattern",
    "extract_all_with_patterns",
    "detect_temporal_context",
    "normalize_value",
    "confidence_from_pattern_match",
    "get_relation_for_type"
]