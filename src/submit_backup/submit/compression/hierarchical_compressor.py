"""
Иерархический компрессор для многоуровневого сжатия
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .compression_models import (
    CompressionConfig, CompressionResult, CompressionLevel, 
    CompressionMethod, TextSegment
)
from .semantic_compressor import SemanticCompressor

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Уровни иерархии сжатия"""
    SENTENCE = "sentence"      # Уровень предложений
    PARAGRAPH = "paragraph"    # Уровень абзацев
    SECTION = "section"        # Уровень разделов
    DOCUMENT = "document"     # Уровень документа


@dataclass
class CompressionHierarchy:
    """Иерархия сжатия"""
    level: HierarchyLevel
    segments: List[TextSegment] = field(default_factory=list)
    compressed_segments: List[TextSegment] = field(default_factory=list)
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalCompressor:
    """Иерархический компрессор для многоуровневого сжатия"""
    
    def __init__(self, config: Optional[CompressionConfig] = None, 
                 base_compressor: Optional[SemanticCompressor] = None):
        """
        Инициализация иерархического компрессора
        
        Args:
            config: Конфигурация сжатия
            base_compressor: Базовый компрессор
        """
        self.config = config or CompressionConfig()
        self.base_compressor = base_compressor or SemanticCompressor(self.config)
        
        # Настройки иерархии
        self.hierarchy_levels = [
            HierarchyLevel.SENTENCE,
            HierarchyLevel.PARAGRAPH,
            HierarchyLevel.SECTION,
            HierarchyLevel.DOCUMENT
        ]
        
        logger.info("HierarchicalCompressor инициализирован")
    
    def compress_hierarchically(self, text: str, 
                               target_level: Optional[HierarchyLevel] = None) -> CompressionResult:
        """
        Выполняет иерархическое сжатие
        
        Args:
            text: Исходный текст
            target_level: Целевой уровень сжатия
            
        Returns:
            Результат сжатия
        """
        target_level = target_level or HierarchyLevel.DOCUMENT
        
        try:
            # Создаем иерархию текста
            hierarchy = self._build_hierarchy(text)
            
            # Сжимаем на каждом уровне
            compressed_hierarchy = self._compress_hierarchy(hierarchy, target_level)
            
            # Собираем финальный результат
            final_text = self._assemble_final_text(compressed_hierarchy, target_level)
            
            # Создаем результат
            result = CompressionResult(
                original_text=text,
                compressed_text=final_text,
                compression_ratio=len(final_text) / len(text) if len(text) > 0 else 1.0,
                method_used=CompressionMethod.HYBRID,
                level_used=self.config.level,
                facts_preserved=self._extract_hierarchy_facts(compressed_hierarchy),
                keywords=self._extract_hierarchy_keywords(compressed_hierarchy),
                metadata={
                    'hierarchy_levels': len(compressed_hierarchy),
                    'target_level': target_level.value,
                    'compression_method': 'hierarchical'
                }
            )
            
            logger.debug(f"Иерархическое сжатие завершено: {len(text)} -> {len(final_text)} символов")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка иерархического сжатия: {e}")
            # Fallback на обычное сжатие
            return self.base_compressor.compress(text)
    
    def _build_hierarchy(self, text: str) -> List[CompressionHierarchy]:
        """Строит иерархию текста"""
        hierarchy = []
        
        # Уровень предложений
        sentences = self._split_into_sentences(text)
        sentence_segments = [
            TextSegment(text=sent, importance=self._calculate_sentence_importance(sent))
            for sent in sentences
        ]
        hierarchy.append(CompressionHierarchy(
            level=HierarchyLevel.SENTENCE,
            segments=sentence_segments
        ))
        
        # Уровень абзацев
        paragraphs = self._split_into_paragraphs(text)
        paragraph_segments = [
            TextSegment(text=para, importance=self._calculate_paragraph_importance(para))
            for para in paragraphs
        ]
        hierarchy.append(CompressionHierarchy(
            level=HierarchyLevel.PARAGRAPH,
            segments=paragraph_segments
        ))
        
        # Уровень разделов (если есть)
        sections = self._split_into_sections(text)
        if sections:
            section_segments = [
                TextSegment(text=sec, importance=self._calculate_section_importance(sec))
                for sec in sections
            ]
            hierarchy.append(CompressionHierarchy(
                level=HierarchyLevel.SECTION,
                segments=section_segments
            ))
        
        # Уровень документа
        doc_segment = TextSegment(text=text, importance=1.0)
        hierarchy.append(CompressionHierarchy(
            level=HierarchyLevel.DOCUMENT,
            segments=[doc_segment]
        ))
        
        return hierarchy
    
    def _compress_hierarchy(self, hierarchy: List[CompressionHierarchy], 
                          target_level: HierarchyLevel) -> List[CompressionHierarchy]:
        """Сжимает иерархию на каждом уровне"""
        compressed_hierarchy = []
        
        for level_hierarchy in hierarchy:
            compressed_segments = []
            
            for segment in level_hierarchy.segments:
                # Определяем уровень сжатия в зависимости от уровня иерархии
                compression_level = self._get_compression_level_for_hierarchy(
                    level_hierarchy.level, target_level
                )
                
                # Сжимаем сегмент
                compressed_result = self.base_compressor.compress(
                    segment.text, 
                    level=compression_level
                )
                
                # Создаем сжатый сегмент
                compressed_segment = TextSegment(
                    text=compressed_result.compressed_text,
                    importance=segment.importance,
                    segment_type=segment.segment_type,
                    metadata={
                        **segment.metadata,
                        'compression_ratio': compressed_result.compression_ratio,
                        'original_length': len(segment.text),
                        'compressed_length': len(compressed_result.compressed_text)
                    }
                )
                
                compressed_segments.append(compressed_segment)
            
            # Обновляем иерархию
            level_hierarchy.compressed_segments = compressed_segments
            level_hierarchy.compression_ratio = self._calculate_level_compression_ratio(
                level_hierarchy.segments, compressed_segments
            )
            
            compressed_hierarchy.append(level_hierarchy)
        
        return compressed_hierarchy
    
    def _assemble_final_text(self, hierarchy: List[CompressionHierarchy], 
                           target_level: HierarchyLevel) -> str:
        """Собирает финальный текст из сжатой иерархии"""
        # Находим нужный уровень
        target_hierarchy = None
        for level_hierarchy in hierarchy:
            if level_hierarchy.level == target_level:
                target_hierarchy = level_hierarchy
                break
        
        if not target_hierarchy:
            # Fallback на документ
            target_hierarchy = hierarchy[-1]
        
        # Собираем текст из сжатых сегментов
        texts = []
        for segment in target_hierarchy.compressed_segments:
            if segment.text.strip():
                texts.append(segment.text.strip())
        
        return ' '.join(texts)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Разбивает текст на абзацы"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Разбивает текст на разделы"""
        # Простая эвристика - ищем заголовки
        import re
        
        # Ищем строки, которые могут быть заголовками
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Проверяем, является ли строка заголовком
            if self._is_header(line):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                current_section.append(line)
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections if len(sections) > 1 else [text]
    
    def _is_header(self, line: str) -> bool:
        """Определяет, является ли строка заголовком"""
        # Простые эвристики для определения заголовков
        if len(line) < 50 and line.isupper():
            return True
        
        if line.startswith(('##', '###', '####')):
            return True
        
        if any(word in line.lower() for word in ['раздел', 'глава', 'часть', 'тема']):
            return True
        
        return False
    
    def _calculate_sentence_importance(self, sentence: str) -> float:
        """Вычисляет важность предложения"""
        if not sentence:
            return 0.0
        
        score = 0.0
        
        # Длина предложения
        word_count = len(sentence.split())
        if word_count > 5:
            score += 0.3
        
        # Ключевые слова
        key_words = ['важно', 'главное', 'основной', 'ключевой', 'принцип', 'метод']
        for word in key_words:
            if word in sentence.lower():
                score += 0.2
        
        # Именованные сущности
        import re
        names = re.findall(r'\b[А-ЯЁA-Z][а-яёa-z]+\b', sentence)
        score += len(names) * 0.1
        
        return min(1.0, score)
    
    def _calculate_paragraph_importance(self, paragraph: str) -> float:
        """Вычисляет важность абзаца"""
        sentences = self._split_into_sentences(paragraph)
        if not sentences:
            return 0.0
        
        # Средняя важность предложений в абзаце
        sentence_importances = [self._calculate_sentence_importance(sent) for sent in sentences]
        return sum(sentence_importances) / len(sentence_importances)
    
    def _calculate_section_importance(self, section: str) -> float:
        """Вычисляет важность раздела"""
        paragraphs = self._split_into_paragraphs(section)
        if not paragraphs:
            return 0.0
        
        # Средняя важность абзацев в разделе
        paragraph_importances = [self._calculate_paragraph_importance(para) for para in paragraphs]
        return sum(paragraph_importances) / len(paragraph_importances)
    
    def _get_compression_level_for_hierarchy(self, hierarchy_level: HierarchyLevel, 
                                           target_level: HierarchyLevel) -> CompressionLevel:
        """Определяет уровень сжатия для уровня иерархии"""
        # Чем выше уровень иерархии, тем более агрессивное сжатие
        level_mapping = {
            HierarchyLevel.SENTENCE: CompressionLevel.LIGHT,
            HierarchyLevel.PARAGRAPH: CompressionLevel.MODERATE,
            HierarchyLevel.SECTION: CompressionLevel.AGGRESSIVE,
            HierarchyLevel.DOCUMENT: CompressionLevel.EXTREME
        }
        
        return level_mapping.get(hierarchy_level, CompressionLevel.MODERATE)
    
    def _calculate_level_compression_ratio(self, original_segments: List[TextSegment], 
                                         compressed_segments: List[TextSegment]) -> float:
        """Вычисляет коэффициент сжатия для уровня"""
        if not original_segments or not compressed_segments:
            return 1.0
        
        original_length = sum(len(seg.text) for seg in original_segments)
        compressed_length = sum(len(seg.text) for seg in compressed_segments)
        
        return compressed_length / original_length if original_length > 0 else 1.0
    
    def _extract_hierarchy_facts(self, hierarchy: List[CompressionHierarchy]) -> List[str]:
        """Извлекает факты из иерархии"""
        facts = []
        
        for level_hierarchy in hierarchy:
            for segment in level_hierarchy.compressed_segments:
                # Извлекаем факты из каждого сегмента
                segment_facts = self.base_compressor._extract_preserved_facts("", segment.text)
                facts.extend(segment_facts)
        
        return list(set(facts))[:10]  # Убираем дубликаты и ограничиваем количество
    
    def _extract_hierarchy_keywords(self, hierarchy: List[CompressionHierarchy]) -> List[str]:
        """Извлекает ключевые слова из иерархии"""
        keywords = []
        
        for level_hierarchy in hierarchy:
            for segment in level_hierarchy.compressed_segments:
                # Извлекаем ключевые слова из каждого сегмента
                segment_keywords = self.base_compressor._extract_keywords(segment.text)
                keywords.extend(segment_keywords)
        
        return list(set(keywords))[:10]  # Убираем дубликаты и ограничиваем количество
