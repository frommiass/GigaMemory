"""
Пакет фильтров для RAG системы
"""

from .message_cleaner import (
    clean_messages,
    clean_user_messages,
    is_personal_message,
    is_technical_content,
    is_copy_paste_content,
    get_message_quality_score
)

from .regex_patterns import (
    COMPILED_PATTERNS,
    PERSONAL_MARKERS,
    COPYPASTE_MARKERS,
    TECH_SIGNS
)

from .session_grouper import (
    SessionGrouper,
    group_messages_by_sessions_simple,
    extract_session_content,
    get_session_summary
)

from .keyword_matcher import (
    KeywordMatcher,
    keyword_matcher,
    find_messages_by_topic,
    find_sessions_by_topic
)

from .relevance_filter import (
    RelevanceFilter,
    relevance_filter,
    find_relevant_sessions
)

__all__ = [
    # Message cleaner
    'clean_messages',
    'clean_user_messages', 
    'is_personal_message',
    'is_technical_content',
    'is_copy_paste_content',
    'get_message_quality_score',
    
    # Regex patterns
    'COMPILED_PATTERNS',
    'PERSONAL_MARKERS',
    'COPYPASTE_MARKERS',
    'TECH_SIGNS',
    
    # Session grouper
    'SessionGrouper',
    'group_messages_by_sessions_simple',
    'extract_session_content',
    'get_session_summary',
    
    # Keyword matcher
    'KeywordMatcher',
    'keyword_matcher',
    'find_messages_by_topic',
    'find_sessions_by_topic',
    
    # Relevance filter
    'RelevanceFilter',
    'relevance_filter',
    'find_relevant_sessions'
]