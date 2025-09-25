# src/submit/__init__.py

"""
GigaMemory - Система долговременной памяти для LLM
Решение для конкурса по созданию глобальной памяти
"""

__version__ = "2.0.0"
__author__ = "GigaMemory Team"

# Главный класс для конкурса
from .model_inference import SubmitModelWithMemory

__all__ = ['SubmitModelWithMemory']