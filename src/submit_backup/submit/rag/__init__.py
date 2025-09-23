"""
Пакет RAG системы
"""

from .config import RAGConfig, DEFAULT_CONFIG
from .engine import RAGEngine, rag_engine, process_question_with_rag
from .interface import RAGInterface, rag_interface, answer_question_with_rag

__all__ = [
    # Config
    'RAGConfig',
    'DEFAULT_CONFIG',
    
    # Engine
    'RAGEngine',
    'rag_engine',
    'process_question_with_rag',
    
    # Interface
    'RAGInterface',
    'rag_interface',
    'answer_question_with_rag'
]