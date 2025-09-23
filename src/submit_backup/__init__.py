"""
GigaMemory - Система долговременной памяти для LLM
Соревнование по созданию глобальной памяти для языковых моделей
"""

__version__ = "2.0.0"
__author__ = "GigaMemory Team"

from .model_inference import SubmitModelWithMemory
from .smart_memory import SmartMemory, SmartMemoryConfig
from .smart_memory_optimized import OptimizedSmartMemory

__all__ = [
    'SubmitModelWithMemory',
    'SmartMemory',
    'SmartMemoryConfig', 
    'OptimizedSmartMemory'
]

# Конфигурация логирования
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())