"""
Модуль для создания эмбеддингов текстов с использованием трансформеров
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModel
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Конфигурация для движка эмбеддингов"""
    model_name: str = "cointegrated/rubert-tiny2"  # Легкая русская модель
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    pooling_strategy: str = "mean"  # mean, max, cls
    cache_dir: Optional[str] = None
    use_cache: bool = True


class EmbeddingEngine:
    """Основной движок для создания эмбеддингов"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Инициализация движка эмбеддингов
        
        Args:
            config: Конфигурация движка
        """
        self.config = config or EmbeddingConfig()
        
        # Загружаем модель и токенизатор
        logger.info(f"Загружаем модель {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        
        # Переносим модель на нужное устройство
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Кэш для эмбеддингов
        self.cache = {} if self.config.use_cache else None
        
        logger.info(f"Модель загружена на {self.device}")
    
    def encode(self, texts: Union[str, List[str]], 
               show_progress: bool = False) -> np.ndarray:
        """
        Кодирует тексты в векторы
        
        Args:
            texts: Текст или список текстов
            show_progress: Показывать прогресс для батчей
            
        Returns:
            Numpy массив эмбеддингов shape (n_texts, embedding_dim)
        """
        # Приводим к списку
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Проверяем кэш
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.config.use_cache:
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.cache:
                    embeddings.append(self.cache[text_hash])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Кодируем некэшированные тексты
        if uncached_texts:
            new_embeddings = self._encode_batch(uncached_texts, show_progress)
            
            # Добавляем в кэш
            if self.config.use_cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    text_hash = self._get_text_hash(text)
                    self.cache[text_hash] = embedding
            
            # Собираем результат в правильном порядке
            if embeddings:
                # Есть кэшированные результаты - нужно объединить
                result = np.zeros((len(texts), new_embeddings.shape[1]))
                cached_idx = 0
                new_idx = 0
                
                for i in range(len(texts)):
                    if i in uncached_indices:
                        result[i] = new_embeddings[new_idx]
                        new_idx += 1
                    else:
                        result[i] = embeddings[cached_idx]
                        cached_idx += 1
                
                embeddings = result
            else:
                embeddings = new_embeddings
        else:
            # Все было в кэше
            embeddings = np.array(embeddings)
        
        # Возвращаем одно значение для одного текста
        if single_text:
            return embeddings[0]
        
        return embeddings
    
    def _encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Кодирует батч текстов
        
        Args:
            texts: Список текстов
            show_progress: Показывать прогресс
            
        Returns:
            Numpy массив эмбеддингов
        """
        all_embeddings = []
        
        # Обрабатываем батчами
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            if show_progress:
                logger.info(f"Обработка батча {i//self.config.batch_size + 1}/{(len(texts)-1)//self.config.batch_size + 1}")
            
            # Токенизация
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Переносим на устройство
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Получаем эмбеддинги
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                # Применяем стратегию пулинга
                embeddings = self._pool_embeddings(
                    outputs.last_hidden_state,
                    encoded['attention_mask']
                )
                
                # Нормализация если нужна
                if self.config.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Конвертируем в numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Объединяем все батчи
        return np.vstack(all_embeddings)
    
    def _pool_embeddings(self, hidden_states: torch.Tensor, 
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Применяет стратегию пулинга к hidden states
        
        Args:
            hidden_states: Выходы модели shape (batch, seq_len, hidden_dim)
            attention_mask: Маска внимания shape (batch, seq_len)
            
        Returns:
            Pooled эмбеддинги shape (batch, hidden_dim)
        """
        if self.config.pooling_strategy == "cls":
            # Берем только CLS токен
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == "max":
            # Max pooling с учетом маски
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), -1e9
            )
            return torch.max(hidden_states, dim=1)[0]
        
        else:  # mean pooling (default)
            # Mean pooling с учетом маски
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
            sum_mask = attention_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
    
    def _get_text_hash(self, text: str) -> str:
        """Получает хэш текста для кэширования"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def encode_with_metadata(self, texts: List[str], 
                            metadata: List[Dict]) -> List[Tuple[np.ndarray, Dict]]:
        """
        Кодирует тексты с сохранением метаданных
        
        Args:
            texts: Список текстов
            metadata: Список метаданных для каждого текста
            
        Returns:
            Список кортежей (embedding, metadata)
        """
        embeddings = self.encode(texts)
        return list(zip(embeddings, metadata))
    
    def save_cache(self, filepath: str):
        """Сохраняет кэш эмбеддингов на диск"""
        if self.cache:
            with open(filepath, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Кэш сохранен: {len(self.cache)} записей")
    
    def load_cache(self, filepath: str):
        """Загружает кэш эмбеддингов с диска"""
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Кэш загружен: {len(self.cache)} записей")
    
    def clear_cache(self):
        """Очищает кэш"""
        if self.cache:
            self.cache.clear()
            logger.info("Кэш очищен")
    
    def get_embedding_dim(self) -> int:
        """Возвращает размерность эмбеддингов"""
        return self.model.config.hidden_size
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между эмбеддингами
        
        Args:
            embedding1: Первый вектор
            embedding2: Второй вектор
            
        Returns:
            Косинусное сходство от -1 до 1
        """
        # Нормализуем если еще не нормализованы
        if not self.config.normalize:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return np.dot(embedding1, embedding2)


class BatchEmbeddingEngine(EmbeddingEngine):
    """Движок с оптимизированной батчевой обработкой"""
    
    def encode_dialogue_sessions(self, sessions: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Кодирует все сессии диалога одним батчем
        
        Args:
            sessions: Словарь {session_id: [messages]}
            
        Returns:
            Словарь {session_id: embedding}
        """
        # Собираем все тексты и их идентификаторы
        all_texts = []
        session_ids = []
        
        for session_id, messages in sessions.items():
            # Объединяем сообщения сессии
            session_text = " ".join(messages)
            all_texts.append(session_text)
            session_ids.append(session_id)
        
        # Кодируем все одним батчем
        embeddings = self.encode(all_texts)
        
        # Возвращаем словарь
        return {
            session_id: embedding 
            for session_id, embedding in zip(session_ids, embeddings)
        }


class CachedEmbeddingEngine(EmbeddingEngine):
    """Движок с персистентным кэшированием"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, 
                 cache_dir: Optional[str] = None):
        """
        Инициализация с поддержкой персистентного кэша
        
        Args:
            config: Конфигурация движка
            cache_dir: Директория для хранения кэша
        """
        super().__init__(config)
        
        self.cache_dir = Path(cache_dir or config.cache_dir or ".embedding_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Загружаем кэш если есть
        cache_file = self.cache_dir / f"{self.config.model_name.replace('/', '_')}.pkl"
        if cache_file.exists():
            self.load_cache(str(cache_file))
    
    def __del__(self):
        """Сохраняем кэш при удалении объекта"""
        if hasattr(self, 'cache_dir') and self.cache:
            cache_file = self.cache_dir / f"{self.config.model_name.replace('/', '_')}.pkl"
            self.save_cache(str(cache_file))



