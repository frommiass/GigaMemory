"""
Мок для библиотеки transformers
"""
from typing import List, Dict, Any


class AutoTokenizer:
    """Мок-класс для AutoTokenizer"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Создает экземпляр токенизатора"""
        return cls()
    
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        """
        Применяет шаблон чата к сообщениям
        Возвращает простой текст без токенизации
        """
        if not messages:
            return ""
        
        result = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                result.append(f"Система: {content}")
            elif role == 'user':
                result.append(f"Пользователь: {content}")
            elif role == 'assistant':
                result.append(f"Ассистент: {content}")
        
        if add_generation_prompt:
            result.append("Ассистент:")
        
        return "\n".join(result)
