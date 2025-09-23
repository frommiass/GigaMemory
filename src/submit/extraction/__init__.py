"""
Пакет для извлечения структурированных фактов из текстов
"""

from .fact_models import (
    Fact,
    FactType,
    FactRelation,
    FactConfidence,
    TemporalFact,
    ConflictingFacts
)

from .fact_extractor import (
    FactExtractor,
    SmartFactExtractor,
    RuleBasedFactExtractor
)

from .fact_database import (
    FactDatabase,
    FactIndex,
    FactConflictResolver,
    FactStats
)

from .fact_patterns import (
    FACT_PATTERNS,
    RELATION_PATTERNS,
    TEMPORAL_PATTERNS,
    get_fact_pattern,
    compile_patterns
)

__all__ = [
    # Models
    'Fact',
    'FactType',
    'FactRelation',
    'FactConfidence',
    'TemporalFact',
    'ConflictingFacts',
    
    # Extractors
    'FactExtractor',
    'SmartFactExtractor',
    'RuleBasedFactExtractor',
    
    # Database
    'FactDatabase',
    'FactIndex',
    'FactConflictResolver',
    'FactStats',
    
    # Patterns
    'FACT_PATTERNS',
    'RELATION_PATTERNS',
    'TEMPORAL_PATTERNS',
    'get_fact_pattern',
    'compile_patterns'
]


