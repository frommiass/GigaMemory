# src/submit/modules/embeddings/embedding_engine.py
"""
Основной движок для создания эмбеддингов
"""
import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Движок для создания эмбеддингов текстов"""
    
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2", 
                 device: str = None, cache_dir: Optional[str] = None):
        """
        Инициализация движка эмбеддингов
        
        Args:
            model_name: Название модели из HuggingFace
            device: Устройство (cuda/cpu/auto)
            cache_dir: Директория для кэша моделей
        """
        self.model_name = model_name
        
        # Определяем устройство
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Инициализация EmbeddingEngine: {model_name} на {self.device}")
        
        # Загружаем модель и токенизатор
        try:
            cache_path = Path(cache_dir) if cache_dir else None
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_path
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_path
            )
            
            # Переносим модель на устройство
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Отключаем градиенты для inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            logger.info(f"Модель {model_name} успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            raise RuntimeError(f"Не удалось загрузить модель {model_name}: {e}")
        
        # Параметры токенизации
        self.max_length = 512
        
        # Внутренний кэш эмбеддингов
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Кодирует один текст в вектор
        
        Args:
            text: Текст для кодирования
            normalize: Нормализовать ли вектор
            
        Returns:
            Вектор эмбеддинга
        """
        # Проверяем кэш
        cache_key = self._get_cache_key(text, normalize)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Токенизация
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Переносим на устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем эмбеддинги
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Mean pooling по токенам
            attention_mask = inputs['attention_mask']
            embeddings = outputs.last_hidden_state
            
            # Маскированное среднее
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # Нормализация если нужна
            if normalize:
                mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            
            # Конвертируем в numpy
            embedding = mean_embeddings.cpu().numpy()[0]
        
        # Сохраняем в кэш
        self.cache[cache_key] = embedding
        
        return embedding
    
    def encode_batch(self, texts: List[str], normalize: bool = True,
                    batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Кодирует батч текстов
        
        Args:
            texts: Список текстов
            normalize: Нормализовать ли векторы
            batch_size: Размер батча
            show_progress: Показывать прогресс
            
        Returns:
            Матрица эмбеддингов
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Обрабатываем батчами
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                logger.info(f"Обработка батча {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Проверяем кэш для каждого текста
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text, normalize)
                if cache_key in self.cache:
                    batch_embeddings.append((j, self.cache[cache_key]))
                    self.cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(j)
                    self.cache_misses += 1
            
            # Кодируем некэшированные тексты
            if uncached_texts:
                # Токенизация батча
                inputs = self.tokenizer(
                    uncached_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Переносим на устройство
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Получаем эмбеддинги
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    batch_embeddings_raw = outputs.last_hidden_state
                    
                    mask_expanded = attention_mask.unsqueeze(-1).expand(
                        batch_embeddings_raw.size()
                    ).float()
                    sum_embeddings = torch.sum(batch_embeddings_raw * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    mean_embeddings = sum_embeddings / sum_mask
                    
                    # Нормализация
                    if normalize:
                        mean_embeddings = torch.nn.functional.normalize(
                            mean_embeddings, p=2, dim=1
                        )
                    
                    # Конвертируем в numpy
                    new_embeddings = mean_embeddings.cpu().numpy()
                
                # Добавляем в кэш и результаты
                for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text, normalize)
                    self.cache[cache_key] = embedding
                    batch_embeddings.append((idx, embedding))
            
            # Сортируем по индексу и добавляем к результатам
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        return np.array(embeddings)
    
    def _get_cache_key(self, text: str, normalize: bool) -> str:
        """Создает ключ для кэша"""
        key = f"{text[:100]}_{normalize}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'model_name': self.model_name,
            'device': str(self.device)
        }
    
    def clear_cache(self):
        """Очищает кэш"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Кэш эмбеддингов очищен")
    
    def warmup(self, sample_texts: List[str] = None):
        """
        Прогрев модели
        
        Args:
            sample_texts: Примеры текстов для прогрева
        """
        if sample_texts is None:
            sample_texts = [
                "Это тестовый текст для прогрева модели.",
                "Модель готова к работе.",
                "Инициализация завершена."
            ]
        
        logger.info(f"Прогрев модели на {len(sample_texts)} примерах")
        _ = self.encode_batch(sample_texts)
        logger.info("Прогрев завершен")