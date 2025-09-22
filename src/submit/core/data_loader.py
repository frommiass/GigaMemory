"""
Центральный модуль для загрузки данных с интегрированной фильтрацией.

Этот модуль обеспечивает централизованную загрузку данных с автоматической фильтрацией
копипаста и технического контента.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from models import Dialog, Message, Session

from .message_filter import MessageFilter


class DataLoader:
    """Центральный класс для загрузки данных с фильтрацией"""
    
    def __init__(self, message_filter: Optional[MessageFilter] = None):
        """
        Инициализация загрузчика данных
        
        Args:
            message_filter: Экземпляр фильтра сообщений (по умолчанию создается новый)
        """
        self.message_filter = message_filter or MessageFilter()
    
    def load_dialogs(self, data_file: str, apply_filtering: bool = True) -> List[Dialog]:
        """
        Загружает диалоги из JSONL файла с опциональной фильтрацией
        
        Args:
            data_file: Путь к JSONL файлу с данными
            apply_filtering: Применять ли фильтрацию копипаста (по умолчанию True)
            
        Returns:
            Список загруженных диалогов
        """
        dialogs = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # ФИЛЬТРАЦИЯ НА САМОМ РАННЕМ ЭТАПЕ - до создания объектов
                        if apply_filtering:
                            data = self._filter_copypaste_from_raw_data(data)
                        
                        dialog = Dialog.from_dict(data)
                        dialogs.append(dialog)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        print(f"Ошибка при загрузке строки {line_num}: {e}")
                        continue
        
        return dialogs
    
    def _filter_copypaste_from_raw_data(self, data: Dict) -> Dict:
        """
        Фильтрует копипаст из сырых JSON данных ДО создания объектов
        
        Args:
            data: Словарь с данными диалога из JSON
            
        Returns:
            Отфильтрованные данные диалога
        """
        from .message_filter import is_copy_paste_content
        
        if 'sessions' not in data:
            return data
        
        filtered_sessions = []
        
        for session_data in data['sessions']:
            if 'messages' not in session_data:
                filtered_sessions.append(session_data)
                continue
            
            filtered_messages = []
            previous_was_copypaste = False
            
            for msg_data in session_data['messages']:
                content = msg_data.get('content', '').strip()
                role = msg_data.get('role', '')
                
                if content:
                    # Проверяем, не является ли сообщение копипастом (для всех ролей)
                    is_copypaste = is_copy_paste_content(content)
                    
                    # Фильтруем копипаст
                    if not is_copypaste:
                        # Сохраняем сообщения, если не копипаст
                        filtered_messages.append(msg_data)
                        previous_was_copypaste = False
                    else:
                        previous_was_copypaste = True
                else:
                    # Сохраняем пустые сообщения
                    filtered_messages.append(msg_data)
                    previous_was_copypaste = False
            
            # Добавляем сессию только если в ней остались сообщения
            if filtered_messages:
                session_data['messages'] = filtered_messages
                filtered_sessions.append(session_data)
        
        # Обновляем данные диалога
        data['sessions'] = filtered_sessions
        return data
    
    def load_dialogs_with_analysis(self, data_file: str, apply_filtering: bool = True) -> tuple[List[Dialog], Dict]:
        """
        Загружает диалоги с анализом фильтрации
        
        Args:
            data_file: Путь к JSONL файлу с данными
            apply_filtering: Применять ли фильтрацию копипаста (по умолчанию True)
            
        Returns:
            Tuple[список_диалогов, анализ_фильтрации]
        """
        # Сначала загружаем без фильтрации для анализа исходных данных
        raw_dialogs = self.load_dialogs(data_file, apply_filtering=False)
        
        # Анализируем исходные данные
        total_dialogs = len(raw_dialogs)
        total_sessions = sum(len(dialog.sessions) for dialog in raw_dialogs)
        total_messages = sum(
            len(session.messages) 
            for dialog in raw_dialogs 
            for session in dialog.sessions
        )
        
        # Применяем фильтрацию, если требуется
        if apply_filtering:
            # Загружаем с фильтрацией (ранняя фильтрация уже применена)
            filtered_dialogs = self.load_dialogs(data_file, apply_filtering=True)
            
            # Анализируем отфильтрованные данные
            filtered_sessions = sum(len(dialog.sessions) for dialog in filtered_dialogs)
            filtered_messages = sum(
                len(session.messages) 
                for dialog in filtered_dialogs 
                for session in dialog.sessions
            )
            
            analysis = {
                'total_dialogs': total_dialogs,
                'filtered_dialogs': len(filtered_dialogs),
                'total_sessions': total_sessions,
                'filtered_sessions': filtered_sessions,
                'total_messages': total_messages,
                'filtered_messages': filtered_messages,
                'dialogs_removed': total_dialogs - len(filtered_dialogs),
                'sessions_removed': total_sessions - filtered_sessions,
                'messages_removed': total_messages - filtered_messages,
                'filtering_applied': True
            }
            
            return filtered_dialogs, analysis
        else:
            analysis = {
                'total_dialogs': total_dialogs,
                'filtered_dialogs': total_dialogs,
                'total_sessions': total_sessions,
                'filtered_sessions': total_sessions,
                'total_messages': total_messages,
                'filtered_messages': total_messages,
                'dialogs_removed': 0,
                'sessions_removed': 0,
                'messages_removed': 0,
                'filtering_applied': False
            }
            
            return raw_dialogs, analysis
    
    def load_dialogs_from_directory(self, directory: str, pattern: str = "*.jsonl", 
                                  apply_filtering: bool = True) -> List[Dialog]:
        """
        Загружает диалоги из всех JSONL файлов в директории
        
        Args:
            directory: Путь к директории с файлами
            pattern: Паттерн для поиска файлов (по умолчанию "*.jsonl")
            apply_filtering: Применять ли фильтрацию копипаста (по умолчанию True)
            
        Returns:
            Список всех загруженных диалогов
        """
        directory_path = Path(directory)
        all_dialogs = []
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file():
                print(f"Загружаем файл: {file_path}")
                dialogs = self.load_dialogs(str(file_path), apply_filtering)
                all_dialogs.extend(dialogs)
        
        return all_dialogs
    
    def save_dialogs(self, dialogs: List[Dialog], output_file: str) -> None:
        """
        Сохраняет диалоги в JSONL файл
        
        Args:
            dialogs: Список диалогов для сохранения
            output_file: Путь к выходному файлу
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for dialog in dialogs:
                # Создаем словарь для сериализации
                dialog_dict = {
                    "id": dialog.id,
                    "question": dialog.question,
                    "sessions": [
                        {
                            "id": session.id,
                            "messages": [
                                {
                                    "role": msg.role,
                                    "content": msg.content,
                                    "session_id": msg.session_id
                                }
                                for msg in session.messages
                            ]
                        }
                        for session in dialog.sessions
                    ]
                }
                json_line = json.dumps(dialog_dict, ensure_ascii=False)
                f.write(json_line + '\n')
    
    def get_loading_stats(self, data_file: str) -> Dict:
        """
        Получает статистику загрузки данных
        
        Args:
            data_file: Путь к JSONL файлу с данными
            
        Returns:
            Словарь со статистикой
        """
        # Загружаем без фильтрации для анализа
        raw_dialogs = self.load_dialogs(data_file, apply_filtering=False)
        
        # Анализируем структуру данных
        stats = {
            'file_path': data_file,
            'total_dialogs': len(raw_dialogs),
            'total_sessions': 0,
            'total_messages': 0,
            'user_messages': 0,
            'assistant_messages': 0,
            'dialogs_with_questions': 0,
            'empty_dialogs': 0,
            'dialogs_by_session_count': {},
            'message_length_stats': {
                'min': float('inf'),
                'max': 0,
                'total': 0,
                'count': 0
            }
        }
        
        for dialog in raw_dialogs:
            # Подсчитываем сессии и сообщения
            dialog_sessions = len(dialog.sessions)
            dialog_messages = sum(len(session.messages) for session in dialog.sessions)
            
            stats['total_sessions'] += dialog_sessions
            stats['total_messages'] += dialog_messages
            
            # Подсчитываем диалоги с вопросами и пустые
            if dialog.question and dialog.question.strip():
                stats['dialogs_with_questions'] += 1
            else:
                stats['empty_dialogs'] += 1
            
            # Группируем по количеству сессий
            if dialog_sessions not in stats['dialogs_by_session_count']:
                stats['dialogs_by_session_count'][dialog_sessions] = 0
            stats['dialogs_by_session_count'][dialog_sessions] += 1
            
            # Анализируем сообщения
            for session in dialog.sessions:
                for msg in session.messages:
                    if msg.role == "user":
                        stats['user_messages'] += 1
                    elif msg.role == "assistant":
                        stats['assistant_messages'] += 1
                    
                    # Статистика длины сообщений
                    msg_len = len(msg.content)
                    stats['message_length_stats']['min'] = min(stats['message_length_stats']['min'], msg_len)
                    stats['message_length_stats']['max'] = max(stats['message_length_stats']['max'], msg_len)
                    stats['message_length_stats']['total'] += msg_len
                    stats['message_length_stats']['count'] += 1
        
        # Вычисляем среднюю длину сообщений
        if stats['message_length_stats']['count'] > 0:
            stats['message_length_stats']['avg'] = stats['message_length_stats']['total'] / stats['message_length_stats']['count']
        else:
            stats['message_length_stats']['avg'] = 0
            stats['message_length_stats']['min'] = 0
        
        return stats


# Глобальный экземпляр для удобства использования
data_loader = DataLoader()


def load_dialogs(data_file: str, apply_filtering: bool = True) -> List[Dialog]:
    """
    Простая функция для загрузки диалогов
    (для обратной совместимости)
    
    Args:
        data_file: Путь к JSONL файлу с данными
        apply_filtering: Применять ли фильтрацию копипаста (по умолчанию True)
        
    Returns:
        Список загруженных диалогов
    """
    return data_loader.load_dialogs(data_file, apply_filtering)
