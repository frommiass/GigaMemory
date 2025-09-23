"""
Пакет промптов для RAG системы
"""

def get_session_marker_prompt(session_id: int) -> str:
    """
    Генерирует маркер сессии для сообщения
    
    Args:
        session_id: Номер сессии
    
    Returns:
        Строка с маркером сессии
    """
    return f"[Сессия {session_id}]"


def get_personal_info_marker() -> str:
    """
    Возвращает маркер для особо важной личной информации
    
    Returns:
        Строка с маркером личной информации
    """
    return "[ЛИЧНОЕ]"
