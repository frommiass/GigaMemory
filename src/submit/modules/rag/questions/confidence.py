"""
Расчет уверенности в классификации вопросов
"""
import math
from typing import Dict, Tuple


def calculate_confidence(score: float, question_length: int, all_scores: Dict[str, float]) -> float:
    """
    Рассчитывает уверенность в классификации
    
    Args:
        score: Счет лучшей темы
        question_length: Длина вопроса в словах
        all_scores: Словарь всех счетов тем
        
    Returns:
        Уверенность от 0.0 до 1.0
    """
    if score <= 0 or question_length <= 0:
        return 0.0
    
    # Базовый confidence на основе счета
    base_confidence = min(score, 1.0)
    
    # Корректировка на основе длины вопроса
    length_factor = _calculate_length_factor(question_length)
    
    # Корректировка на основе конкуренции между темами
    competition_factor = _calculate_competition_factor(all_scores)
    
    # Итоговый confidence
    confidence = base_confidence * length_factor * competition_factor
    
    return min(confidence, 1.0)


def _calculate_length_factor(question_length: int) -> float:
    """
    Рассчитывает фактор длины вопроса
    
    Args:
        question_length: Длина вопроса в словах
        
    Returns:
        Фактор от 0.5 до 1.0
    """
    if question_length <= 2:
        return 0.5  # Очень короткие вопросы менее надежны
    elif question_length <= 5:
        return 0.7  # Короткие вопросы
    elif question_length <= 10:
        return 0.9  # Средние вопросы
    else:
        return 1.0  # Длинные вопросы более надежны


def _calculate_competition_factor(all_scores: Dict[str, float]) -> float:
    """
    Рассчитывает фактор конкуренции между темами
    
    Args:
        all_scores: Словарь всех счетов тем
        
    Returns:
        Фактор от 0.5 до 1.0
    """
    if len(all_scores) <= 1:
        return 1.0  # Нет конкуренции
    
    # Сортируем счета по убыванию
    sorted_scores = sorted(all_scores.values(), reverse=True)
    
    if len(sorted_scores) < 2:
        return 1.0
    
    best_score = sorted_scores[0]
    second_best_score = sorted_scores[1]
    
    # Если вторая тема сильно отстает, confidence высокий
    if second_best_score == 0:
        return 1.0
    
    # Рассчитываем соотношение лучшей и второй темы
    ratio = best_score / second_best_score
    
    if ratio >= 2.0:
        return 1.0  # Четкое лидерство
    elif ratio >= 1.5:
        return 0.9  # Хорошее лидерство
    elif ratio >= 1.2:
        return 0.8  # Умеренное лидерство
    else:
        return 0.6  # Слабое лидерство


def calculate_confidence_with_threshold(score: float, question_length: int, 
                                      all_scores: Dict[str, float], 
                                      threshold: float = 0.7) -> Tuple[float, bool]:
    """
    Рассчитывает confidence и проверяет, превышает ли он порог
    
    Args:
        score: Счет лучшей темы
        question_length: Длина вопроса в словах
        all_scores: Словарь всех счетов тем
        threshold: Пороговое значение confidence
        
    Returns:
        Tuple[confidence, exceeds_threshold]
    """
    confidence = calculate_confidence(score, question_length, all_scores)
    exceeds_threshold = confidence >= threshold
    
    return confidence, exceeds_threshold


def get_confidence_level(confidence: float) -> str:
    """
    Получает текстовое описание уровня уверенности
    
    Args:
        confidence: Значение confidence от 0.0 до 1.0
        
    Returns:
        Текстовое описание уровня
    """
    if confidence >= 0.9:
        return "очень высокая"
    elif confidence >= 0.8:
        return "высокая"
    elif confidence >= 0.7:
        return "средняя"
    elif confidence >= 0.5:
        return "низкая"
    else:
        return "очень низкая"
