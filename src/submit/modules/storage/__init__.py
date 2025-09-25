# src/submit/storage/__init__.py
"""
Пакет хранилищ для RAG системы
"""

from .memory_storage import MemoryStorage
from .memory_models import MemoryRegistry, MemoryStats, CacheStats, MemoryEntry, CacheEntry
from .message_filter import MessageFilter, message_filter
from .module import StorageModule

# Объединенный SessionGrouper (вместо отдельных SessionManager и старого SessionGrouper)
from .filters.session_grouper import (
    SessionGrouper,
    group_messages_by_sessions_simple,
    extract_session_content,
    extract_session_content_for_search,
    get_session_summary
)

__all__ = [
    # Main storage classes
    'StorageModule',
    'MemoryStorage',
    'MessageFilter',
    'SessionGrouper',
    
    # Memory models
    'MemoryRegistry',
    'MemoryStats',
    'CacheStats',
    'MemoryEntry',
    'CacheEntry',
    
    # Helper functions
    'group_messages_by_sessions_simple',
    'extract_session_content',
    'extract_session_content_for_search',
    'get_session_summary',
    
    # Global instances
    'message_filter'
]