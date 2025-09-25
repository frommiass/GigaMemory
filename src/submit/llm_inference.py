from core.interfaces import IModelInference, ProcessingResult
"""
Модуль для инференса и работы с языковой моделью GigaChat
"""
from dataclasses import asdict
from typing import List
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Заглушки для случая отсутствия vllm
    class LLM:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            return [type('Output', (), {'outputs': [type('Output', (), {'text': 'Заглушка ответа'})]})()]
    
    class SamplingParams:
        def __init__(self, *args, **kwargs):
            pass
from models import Message


class ModelInference:
    """Класс для работы с языковой моделью"""
    
    def __init__(self, model_path: str):
        """
        Инициализация модели
        
        Args:
            model_path: Путь к модели
        """
        self.model_path = model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            if VLLM_AVAILABLE:
                self.model = LLM(
                    model=self.model_path, 
                    trust_remote_code=True
                )
                self.sampling_params = SamplingParams(
                    temperature=0.0, 
                    max_tokens=100, 
                    seed=42, 
                    truncate_prompt_tokens=131072
                )
            else:
                self.model = LLM()
                self.sampling_params = SamplingParams()
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели {self.model_path}: {str(e)}")
    
    
    def generate(self, messages: List[Message]) -> ProcessingResult:
        """Реализация метода интерфейса IModelInference"""
        try:
            result = self.inference(messages)
            return ProcessingResult(
                success=True,
                data=result,
                metadata={
                    'model_path': self.model_path,
                    'messages_count': len(messages)
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data="Ошибка генерации.",
                metadata={'error': str(e)},
                error=str(e)
            )

def inference(self, messages: List[Message]) -> str:
        """
        Выполняет инференс модели
        
        Args:
            messages: Список сообщений для модели
            
        Returns:
            Ответ модели
        """
        try:
            msg_dicts = [asdict(m) for m in messages]
            input_tensor = self.tokenizer.apply_chat_template(
                msg_dicts,
                add_generation_prompt=True,
            )
            outputs = self.model.generate(
                prompt_token_ids=input_tensor, 
                sampling_params=self.sampling_params, 
                use_tqdm=False
            )
            result = outputs[0].outputs[0].text
            return result.strip()
        except Exception as e:
            return f"Ошибка при инференсе локальной модели: {str(e)}"
