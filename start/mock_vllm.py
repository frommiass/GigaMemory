"""
Мок для библиотеки vllm
"""
from typing import List, Dict, Any


class SamplingParams:
    """Мок-класс для параметров семплирования"""
    
    def __init__(self, temperature: float = 0.0, max_tokens: int = 100, 
                 seed: int = 42, truncate_prompt_tokens: int = 131072, **kwargs):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.truncate_prompt_tokens = truncate_prompt_tokens


class Output:
    """Мок-класс для вывода генерации"""
    
    def __init__(self, text: str):
        self.text = text


class RequestOutput:
    """Мок-класс для результата запроса"""
    
    def __init__(self, outputs: List[Output]):
        self.outputs = outputs


class LLM:
    """Мок-класс для языковой модели"""
    
    def __init__(self, model: str, trust_remote_code: bool = True, **kwargs):
        self.model_path = model
        self.trust_remote_code = trust_remote_code
    
    def generate(self, prompt_token_ids: str, sampling_params: SamplingParams, 
                 use_tqdm: bool = False) -> List[RequestOutput]:
        """
        Генерирует ответ на основе промпта
        Возвращает мок-ответы для тестирования
        """
        # Разные промпты для каждого диалога на основе содержимого
        prompt_text = str(prompt_token_ids).lower()
        
        # Диалог 1: спорт
        if "каким спортом" in prompt_text:
            response = "Вы играете в футбол, пинг-понг"
        # Диалог 2: работа  
        elif "кем я работаю" in prompt_text:
            response = "Вы работаете сварщиком"
        # Диалог 3: собака
        elif "какая порода" in prompt_text:
            response = "Мальтийская болонка"
        # Диалог 4: сигареты
        elif "сигареты" in prompt_text:
            response = "У меня нет такой информации"
        else:
            response = "Не удалось найти информацию"
        
        return [RequestOutput([Output(response)])]
