# modules/compression/module.py
from core.interfaces import ICompressor, ProcessingResult
from typing import Dict, Any, List

class CompressionModule(ICompressor):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        from .compressor import SemanticCompressor, HierarchicalCompressor
        from .strategies import CompressionStrategy
        
        self.semantic = SemanticCompressor(config)
        self.hierarchical = HierarchicalCompressor(config)
        self.strategy = CompressionStrategy(config.get('method', 'hybrid'))
        
        self.optimizer = None
        self.stats = {
            'total_compressed': 0,
            'total_saved_chars': 0,
            'compression_time': 0
        }
    
    def set_optimizer(self, optimizer):
        """Устанавливает оптимизатор"""
        self.optimizer = optimizer
        if optimizer:
            optimizer.optimize_for_text()  # Включаем оптимизацию для текста
    
    def compress_text(self, text: str, level: str = "moderate") -> ProcessingResult:
        """Сжимает текст"""
        try:
            # Кэшируем результаты сжатия
            if self.optimizer:
                cache_key = f"compress_{hash(text)}_{level}"
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True, 'level': level}
                    )
            
            # Выбираем метод сжатия
            if level == "light":
                compressed = self.semantic.compress_light(text)
            elif level == "heavy":
                compressed = self.hierarchical.compress_heavy(text)
            else:
                compressed = self.strategy.compress_adaptive(text)
            
            # Обновляем статистику
            self.stats['total_compressed'] += 1
            self.stats['total_saved_chars'] += len(text) - len(compressed)
            
            # Кэшируем
            if self.optimizer:
                self.optimizer.cache_put(cache_key, compressed, ttl=7200)
            
            return ProcessingResult(
                success=True,
                data=compressed,
                metadata={
                    'original_length': len(text),
                    'compressed_length': len(compressed),
                    'ratio': len(compressed) / len(text) if text else 0
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=text,
                error=str(e)
            )
    
    def compress_sessions(self, sessions: Dict[str, str]) -> ProcessingResult:
        """Сжимает множество сессий"""
        try:
            compressed = {}
            total_original = 0
            total_compressed = 0
            
            # Батчевая обработка если есть оптимизатор
            if self.optimizer and len(sessions) > 5:
                tasks = [
                    {'id': sid, 'text': text, 'priority': len(text)}
                    for sid, text in sessions.items()
                ]
                
                batch_result = self.optimizer.batch_process_priority(
                    tasks,
                    lambda t: self.compress_text(t['text']).data
                )
                
                if batch_result.success:
                    for i, (sid, _) in enumerate(sessions.items()):
                        compressed[sid] = batch_result.data[i]
                        total_original += len(sessions[sid])
                        total_compressed += len(batch_result.data[i])
            else:
                # Обычное сжатие
                for sid, text in sessions.items():
                    result = self.compress_text(text)
                    compressed[sid] = result.data
                    total_original += len(text)
                    total_compressed += len(result.data)
            
            return ProcessingResult(
                success=True,
                data=compressed,
                metadata={
                    'sessions_count': len(sessions),
                    'total_ratio': total_compressed / total_original if total_original > 0 else 0
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=sessions,
                error=str(e)
            )
    
    def get_compression_stats(self) -> ProcessingResult:
        """Получает статистику сжатия"""
        return ProcessingResult(
            success=True,
            data=self.stats.copy(),
            metadata={'timestamp': __import__('datetime').datetime.now().isoformat()}
        )
    
    # Дополнительный метод для RAG
    def compress_for_context(self, text: str, max_length: int, preserve_keywords: List[str] = None) -> str:
        """Специальное сжатие для контекста вопроса"""
        if len(text) <= max_length:
            return text
        
        if preserve_keywords:
            # Сохраняем предложения с ключевыми словами
            sentences = text.split('. ')
            important = []
            other = []
            
            for sent in sentences:
                if any(kw.lower() in sent.lower() for kw in preserve_keywords):
                    important.append(sent)
                else:
                    other.append(sent)
            
            # Собираем результат
            result = '. '.join(important)
            remaining_space = max_length - len(result)
            
            if remaining_space > 100:
                for sent in other:
                    if len(result) + len(sent) + 2 < max_length:
                        result += '. ' + sent
                    else:
                        break
            
            return result[:max_length]
        else:
            # Адаптивное сжатие
            return self.strategy.compress_to_length(text, max_length)