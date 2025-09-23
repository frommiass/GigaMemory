"""
Мок для библиотеки transformers
"""
from typing import List, Dict, Any
import torch


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
    
    def __call__(self, texts: List[str], padding: bool = True, truncation: bool = True,
                 max_length: int = 512, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        seq_len = min(max_length, 16)
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _ModelConfig:
    def __init__(self, hidden_size: int = 384):
        self.hidden_size = hidden_size


class _ModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class AutoModel:
    """Мок-класс для AutoModel"""
    
    def __init__(self, hidden_size: int = 384):
        self.config = _ModelConfig(hidden_size=hidden_size)
        self._device = torch.device("cpu")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        return cls()
    
    def to(self, device):
        self._device = device
        return self
    
    def eval(self):
        return self
    
    def __call__(self, **encoded) -> _ModelOutput:
        input_ids: torch.Tensor = encoded.get("input_ids")
        if input_ids is None:
            # Fallback размеры
            batch, seq_len = 1, 16
        else:
            batch, seq_len = input_ids.shape
        hidden = self.config.hidden_size
        last_hidden = torch.randn((batch, seq_len, hidden), device=self._device)
        return _ModelOutput(last_hidden_state=last_hidden)
