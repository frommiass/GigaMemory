"""
Обновленная модель с семантическим сжатием
"""
from typing import List

from models import Message
from submit_interface import ModelWithMemory

from .storage import MemoryStorage
from .rag.compressed_rag_engine import CompressedRAGEngine, CompressedRAGConfig
from .compression.compression_models import CompressionLevel, CompressionMethod
from .llm_inference import ModelInference


class SubmitModelWithMemoryV2(ModelWithMemory):
    """
    Обновленный класс с семантическим сжатием
    """

    def __init__(self, model_path: str, enable_compression: bool = True) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        
        # Создаем конфигурацию с сжатием
        config = CompressedRAGConfig(
            enable_compression=enable_compression,
            compression_level=CompressionLevel.MODERATE,
            compression_method=CompressionMethod.HYBRID,
            compression_target_ratio=0.3,
            max_relevant_sessions=5
        )
        
        # Используем сжимающий RAG движок
        self.rag_interface = CompressedRAGEngine(config, self.model_inference)

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Фильтрует, сохраняет и сжимает личную информацию из сообщений
        """
        # Группируем сообщения по сессиям (используем существующую логику)
        session_messages = {}
        
        for msg in messages:
            if msg.role == "user":
                # Проверяем кэш для ускорения
                cached_result = self.storage.check_cache(msg.content)
                
                if cached_result is None:
                    from .core.message_filter import is_personal_message
                    filter_result = is_personal_message(msg.content)
                    self.storage.add_to_cache(msg.content, filter_result)
                    cached_result = filter_result
                
                if cached_result:
                    # Используем session_id из сообщения, если он есть
                    session_id = msg.session_id if msg.session_id else str(self.storage.increment_session(dialogue_id))
                    
                    # Регистрируем сессию
                    if msg.session_id:
                        self.storage.register_session(dialogue_id, session_id)
                    
                    # Группируем по session_id
                    if session_id not in session_messages:
                        session_messages[session_id] = []
                    
                    session_messages[session_id].append(msg)
        
        # Сохраняем сообщения в память (БЕЗ предварительного склеивания)
        processed_messages = []
        for session_id, msgs in session_messages.items():
            for msg in msgs:
                processed_msg = Message(
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id
                )
                processed_messages.append(processed_msg)
        
        self.storage.add_to_memory(dialogue_id, processed_messages)

    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память диалога и кэш сжатия
        """
        self.storage.clear_dialogue_memory(dialogue_id)
        
        # Очищаем кэш сжатия
        self.rag_interface.clear_compression_cache()
        
        if self.storage.get_cache_size() > 1000:
            self.storage.clear_all_cache()

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """
        Генерирует ответ на вопрос используя сжимающую RAG систему
        """
        # Получаем все сообщения из памяти
        memory = self.storage.get_memory(dialogue_id)
        
        if not memory:
            return "У меня нет информации для ответа на этот вопрос."
        
        # Используем сжимающую RAG систему для генерации промпта
        rag_prompt, metadata = self.rag_interface.process_question(question, dialogue_id, memory)
        
        # Создаем контекст для модели
        context_with_memory = [Message('system', rag_prompt)]
        
        # Генерируем ответ через модель
        answer = self.model_inference.inference(context_with_memory)
        return answer
    
    def get_compression_stats(self, dialogue_id: str) -> dict:
        """
        Получает статистику сжатия для диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь со статистикой сжатия
        """
        memory = self.storage.get_memory(dialogue_id)
        if not memory:
            return {'error': 'Нет данных в памяти'}
        
        # Получаем статистику сжатия от RAG движка
        compression_stats = self.rag_interface.get_compression_stats()
        
        # Добавляем общую статистику
        dialogue_stats = {
            'dialogue_id': dialogue_id,
            'memory_messages': len(memory),
            'sessions_count': len(set(msg.session_id for msg in memory if msg.session_id)),
        }
        
        return {**dialogue_stats, **compression_stats}
    
    def analyze_compression(self, dialogue_id: str, question: str) -> dict:
        """
        Анализирует эффективность сжатия для вопроса
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            
        Returns:
            Словарь с анализом сжатия
        """
        memory = self.storage.get_memory(dialogue_id)
        if not memory:
            return {'error': 'Нет данных в памяти'}
        
        # Получаем метаданные обработки вопроса
        rag_prompt, metadata = self.rag_interface.process_question(question, dialogue_id, memory)
        
        # Анализируем эффективность сжатия
        analysis = {
            'question': question,
            'dialogue_id': dialogue_id,
            'strategy_used': metadata.get('strategy', 'unknown'),
            'compression_enabled': metadata.get('compression_enabled', False),
            'total_sessions': metadata.get('total_sessions', 0),
            'selected_sessions': metadata.get('selected_sessions', 0),
            'memory_length': metadata.get('memory_length', 0),
            'compression_efficiency': 0.0
        }
        
        # Рассчитываем эффективность сжатия
        if analysis['total_sessions'] > 0:
            analysis['compression_efficiency'] = analysis['selected_sessions'] / analysis['total_sessions']
        
        # Добавляем статистику компрессора
        compressor_stats = self.rag_interface.get_compression_stats()
        analysis['compressor_stats'] = compressor_stats.get('compressor_stats', {})
        
        return analysis
    
    def optimize_compression_settings(self, dialogue_id: str) -> dict:
        """
        Предлагает оптимальные настройки сжатия на основе анализа диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь с рекомендациями по настройке сжатия
        """
        memory = self.storage.get_memory(dialogue_id)
        if not memory:
            return {'error': 'Нет данных в памяти'}
        
        # Анализируем характеристики диалога
        total_messages = len(memory)
        sessions = set(msg.session_id for msg in memory if msg.session_id)
        total_sessions = len(sessions)
        
        # Анализируем длину сообщений
        message_lengths = [len(msg.content) for msg in memory if msg.role == "user"]
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        max_message_length = max(message_lengths) if message_lengths else 0
        
        # Рекомендации по настройке
        recommendations = {
            'dialogue_id': dialogue_id,
            'analysis': {
                'total_messages': total_messages,
                'total_sessions': total_sessions,
                'avg_message_length': avg_message_length,
                'max_message_length': max_message_length
            },
            'recommendations': {}
        }
        
        # Рекомендации по уровню сжатия
        if avg_message_length > 500:
            recommendations['recommendations']['compression_level'] = 'AGGRESSIVE'
            recommendations['recommendations']['reason'] = 'Длинные сообщения требуют агрессивного сжатия'
        elif avg_message_length > 200:
            recommendations['recommendations']['compression_level'] = 'MODERATE'
            recommendations['recommendations']['reason'] = 'Средняя длина сообщений подходит для умеренного сжатия'
        else:
            recommendations['recommendations']['compression_level'] = 'LIGHT'
            recommendations['recommendations']['reason'] = 'Короткие сообщения требуют легкого сжатия'
        
        # Рекомендации по методу сжатия
        if total_sessions > 20:
            recommendations['recommendations']['compression_method'] = 'HYBRID'
            recommendations['recommendations']['hierarchical_threshold'] = 10
            recommendations['recommendations']['reason'] += '; Много сессий - используйте гибридный метод с иерархией'
        else:
            recommendations['recommendations']['compression_method'] = 'EXTRACTIVE'
            recommendations['recommendations']['reason'] += '; Мало сессий - достаточно экстрактивного метода'
        
        # Рекомендации по целевому коэффициенту
        if max_message_length > 1000:
            recommendations['recommendations']['target_ratio'] = 0.2
            recommendations['recommendations']['reason'] += '; Очень длинные сообщения требуют сильного сжатия'
        elif avg_message_length > 300:
            recommendations['recommendations']['target_ratio'] = 0.3
            recommendations['recommendations']['reason'] += '; Средняя длина требует умеренного сжатия'
        else:
            recommendations['recommendations']['target_ratio'] = 0.5
            recommendations['recommendations']['reason'] += '; Короткие сообщения требуют легкого сжатия'
        
        return recommendations
    
    def get_memory_summary(self, dialogue_id: str) -> dict:
        """
        Получает краткое резюме памяти диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь с резюме памяти
        """
        memory = self.storage.get_memory(dialogue_id)
        if not memory:
            return {'error': 'Нет данных в памяти'}
        
        # Группируем по сессиям
        sessions = {}
        for msg in memory:
            if msg.role == "user" and msg.session_id:
                if msg.session_id not in sessions:
                    sessions[msg.session_id] = []
                sessions[msg.session_id].append(msg.content)
        
        # Создаем резюме
        summary = {
            'dialogue_id': dialogue_id,
            'total_messages': len(memory),
            'total_sessions': len(sessions),
            'sessions_summary': []
        }
        
        # Резюме по сессиям
        for session_id, messages in sessions.items():
            session_summary = {
                'session_id': session_id,
                'message_count': len(messages),
                'total_length': sum(len(msg) for msg in messages),
                'avg_length': sum(len(msg) for msg in messages) / len(messages) if messages else 0,
                'sample_message': messages[0][:100] + '...' if messages else ''
            }
            summary['sessions_summary'].append(session_summary)
        
        return summary
