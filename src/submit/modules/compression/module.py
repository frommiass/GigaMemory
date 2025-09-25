# modules/compression/module.py

from typing import Dict, Any
from ...core.interfaces import ICompressor, ProcessingResult

from .compression_models import CompressionConfig, CompressionLevel, CompressionMethod
from .semantic_compressor import SemanticCompressor, HybridCompressor
from .hierarchical_compressor import HierarchicalCompressor


class CompressionModule(ICompressor):
    """Модуль семантического сжатия"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Создаем конфигурацию сжатия
        comp_config = CompressionConfig(
            level=CompressionLevel(config.get('level', 'moderate')),
            method=CompressionMethod(config.get('method', 'hybrid')),
            target_ratio=config.get('target_ratio', 0.3),
            use_cache=config.get('use_cache', True)
        )
        
        # Выбираем компрессор
        if config.get('use_hierarchical', False):
            self.compressor = HierarchicalCompressor(comp_config)
        elif config.get('method') == 'hybrid':
            self.compressor = HybridCompressor(comp_config)
        else:
            self.compressor = SemanticCompressor(comp_config)
        
        self.stats = self.compressor.stats
    
    def compress_text(self, text: str, level: str = "moderate") -> ProcessingResult:
        """Сжимает текст"""
        try:
            compression_level = CompressionLevel(level)
            result = self.compressor.compress(text, level=compression_level)
            
            return ProcessingResult(
                success=True,
                data=result.compressed_text,
                metadata={
                    'original_length': result.original_length,
                    'compressed_length': result.compressed_length,
                    'compression_ratio': result.compression_ratio,
                    'method': result.method_used.value
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=text,  # Возвращаем оригинал при ошибке
                metadata={},
                error=str(e)
            )
    
    def compress_sessions(self, sessions: Dict[str, str]) -> ProcessingResult:
        """Сжимает множество сессий"""
        try:
            compressed_sessions = {}
            total_original = 0
            total_compressed = 0
            
            for session_id, text in sessions.items():
                result = self.compressor.compress(text)
                compressed_sessions[session_id] = result.compressed_text
                total_original += result.original_length
                total_compressed += result.compressed_length
            
            ratio = total_compressed / total_original if total_original > 0 else 1.0
            
            return ProcessingResult(
                success=True,
                data=compressed_sessions,
                metadata={
                    'sessions_count': len(sessions),
                    'total_original': total_original,
                    'total_compressed': total_compressed,
                    'ratio': ratio
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=sessions,  # Возвращаем оригинал при ошибке
                metadata={},
                error=str(e)
            )
    
    def get_compression_stats(self) -> ProcessingResult:
        """Получает статистику сжатия"""
        try:
            stats = self.stats.to_dict()
            
            return ProcessingResult(
                success=True,
                data=stats,
                metadata={}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )