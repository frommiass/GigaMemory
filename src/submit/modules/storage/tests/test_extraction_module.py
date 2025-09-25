# tests/test_extraction_module.py
"""
Тесты для проверки работы улучшенного модуля извлечения
"""

import unittest
import sys
import os
from datetime import datetime

# Добавляем пути для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'submit'))

from modules.extraction.module import ExtractionModule
from core.interfaces import ProcessingResult


class TestExtractionModule(unittest.TestCase):
    """Тесты для модуля извлечения"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.config = {
            'use_rules': True,
            'use_llm': False,
            'min_confidence': 0.5,
            'filter_copypaste': True,
            'max_text_length': 10000,
            'conflict_strategy': 'latest'
        }
        
        self.module = ExtractionModule(self.config)
        self.test_session_id = "test_session_001"
        self.test_dialogue_id = "test_dialogue_001"
    
    def test_extract_basic_facts(self):
        """Тест извлечения базовых фактов"""
        text = "Привет! Меня зовут Александр, мне 28 лет. Я живу в Москве и работаю программистом в Яндексе."
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        result = self.module.extract_facts(text, context)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.data), 0)
        
        # Проверяем извлеченные факты
        facts = result.data
        fact_objects = [f.object for f in facts]
        
        self.assertIn("Александр", fact_objects)
        self.assertIn("28", fact_objects)
        self.assertIn("Москве", fact_objects)
        self.assertIn("программистом", fact_objects)
        self.assertIn("Яндексе", fact_objects)
    
    def test_extract_info_updating(self):
        """Тест извлечения обновлений информации"""
        # Сначала добавляем базовую информацию
        text1 = "Меня зовут Мария, мне 25 лет. Я не замужем, работаю учителем."
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        result1 = self.module.extract_facts(text1, context)
        self.assertTrue(result1.success)
        
        # Теперь обновляем информацию
        text2 = "Кстати, я недавно вышла замуж и теперь мне уже 26 лет. Перешла работать в Google."
        
        result2 = self.module.extract_facts(text2, context)
        self.assertTrue(result2.success)
        
        # Проверяем, что обновления были обнаружены
        self.assertTrue(result2.metadata.get('is_info_update', False))
        
        # Проверяем обновленные факты
        facts = result2.data
        fact_objects = [f.object for f in facts]
        
        self.assertIn("26", fact_objects)  # Новый возраст
        self.assertIn("замужем", fact_objects)  # Новый статус
        self.assertIn("Google", fact_objects)  # Новая компания
    
    def test_get_critical_facts(self):
        """Тест получения критических фактов"""
        # Добавляем факты
        text = "Я Петр Иванов, 35 лет. Живу в Санкт-Петербурге, работаю врачом в больнице. Женат, двое детей."
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        self.module.extract_facts(text, context)
        
        # Получаем критические факты
        critical = self.module.get_critical_facts(self.test_dialogue_id)
        
        self.assertIsNotNone(critical)
        self.assertIn('name', critical)
        self.assertIn('age', critical)
        self.assertIn('location', critical)
        self.assertIn('occupation', critical)
        self.assertIn('marital_status', critical)
        
        self.assertEqual(critical.get('name'), 'Петр')
        self.assertEqual(critical.get('age'), '35')
        self.assertEqual(critical.get('marital_status'), 'женат')
    
    def test_caching(self):
        """Тест кэширования фактов"""
        text = "Меня зовут Анна, работаю дизайнером."
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        # Первое извлечение
        result1 = self.module.extract_facts(text, context)
        self.assertTrue(result1.success)
        self.assertFalse(result1.metadata.get('from_cache', False))
        
        # Второе извлечение того же текста (должно быть из кэша)
        # Но только если установлен optimizer
        result2 = self.module.extract_facts(text, context)
        self.assertTrue(result2.success)
        # Без optimizer кэширование не работает
        self.assertFalse(result2.metadata.get('from_cache', False))
    
    def test_complex_family_facts(self):
        """Тест извлечения сложных семейных фактов"""
        text = """
        У меня большая семья. Жену зовут Елена, ей 30 лет. 
        У нас трое детей: старший сын Максим, средняя дочь Софья и младший сын Артем.
        Еще у нас есть кот Барсик и собака породы лабрадор по кличке Рекс.
        """
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        result = self.module.extract_facts(text, context)
        self.assertTrue(result.success)
        
        facts = result.data
        fact_objects = [f.object for f in facts]
        
        # Проверяем семейные факты
        self.assertIn("Елена", fact_objects)  # Жена
        self.assertIn("Максим", fact_objects)  # Сын
        self.assertIn("Софья", fact_objects)  # Дочь
        self.assertIn("Артем", fact_objects)  # Сын
        
        # Проверяем питомцев
        self.assertIn("Барсик", fact_objects)  # Кот
        self.assertIn("Рекс", fact_objects)  # Собака
        self.assertIn("лабрадор", fact_objects)  # Порода
    
    def test_temporal_facts(self):
        """Тест временных фактов"""
        text = "Раньше я работал в банке, но теперь я фрилансер. Был женат, развелся год назад."
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        result = self.module.extract_facts(text, context)
        self.assertTrue(result.success)
        
        facts = result.data
        
        # Проверяем, что есть временные факты
        temporal_facts = [f for f in facts if hasattr(f, 'is_current')]
        self.assertGreater(len(temporal_facts), 0)
        
        # Проверяем текущие факты
        current_facts = [f for f in temporal_facts if f.is_current]
        fact_objects = [f.object for f in current_facts]
        
        self.assertIn("фрилансер", fact_objects)
        self.assertIn("разведен", fact_objects)
    
    def test_copypaste_filter(self):
        """Тест фильтрации копипаста"""
        # Текст без личных маркеров (вероятный копипаст)
        copypaste_text = """
        Москва — столица России, город федерального значения, 
        административный центр Центрального федерального округа и центр Московской области, 
        в состав которой не входит. Крупнейший по численности населения город России 
        и её субъект — 13 010 112 человек (2023), самый населённый из городов, 
        полностью расположенных в Европе, занимает 22-е место среди городов мира по численности населения.
        """ * 10  # Длинный текст
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        result = self.module.extract_facts(copypaste_text, context)
        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 0)  # Факты не извлечены
        self.assertEqual(result.metadata.get('filtered'), 'copypaste')
    
    def test_get_user_profile(self):
        """Тест построения профиля пользователя"""
        # Добавляем разнообразные факты
        texts = [
            "Меня зовут Игорь Петров, мне 42 года.",
            "Живу в Екатеринбурге, работаю инженером в Уралмаше.",
            "Женат на Ольге, у нас двое детей.",
            "Увлекаюсь фотографией и горными лыжами.",
            "Окончил УрФУ по специальности механика.",
            "У меня есть кот Мурзик."
        ]
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        for text in texts:
            self.module.extract_facts(text, context)
        
        # Получаем профиль
        result = self.module.get_user_profile(self.test_dialogue_id)
        self.assertTrue(result.success)
        
        profile = result.data
        self.assertIn('critical_facts', profile)
        self.assertIn('personal', profile)
        self.assertIn('family', profile)
        self.assertIn('work', profile)
        self.assertIn('education', profile)
        self.assertIn('hobbies', profile)
        self.assertIn('pets', profile)
        
        # Проверяем критические факты
        critical = profile['critical_facts']
        self.assertEqual(critical.get('name'), 'Игорь')
        self.assertEqual(critical.get('age'), '42')
        self.assertEqual(critical.get('location'), 'Екатеринбурге')
    
    def test_query_facts(self):
        """Тест поиска фактов"""
        # Добавляем факты
        text = "Я Сергей, работаю программистом в Сбере. Люблю кофе и jazz музыку."
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        self.module.extract_facts(text, context)
        
        # Поиск по запросу
        result = self.module.query_facts(self.test_dialogue_id, "работа")
        self.assertTrue(result.success)
        self.assertGreater(len(result.data), 0)
        
        # Проверяем, что нашли факты о работе
        facts = result.data
        fact_objects = [f.object for f in facts]
        self.assertIn("программистом", fact_objects)
    
    def test_stats(self):
        """Тест статистики модуля"""
        # Извлекаем несколько фактов
        texts = [
            "Меня зовут Виктор, 30 лет.",
            "Работаю в IT компании.",
            "У меня есть собака."
        ]
        
        context = {
            'session_id': self.test_session_id,
            'dialogue_id': self.test_dialogue_id
        }
        
        for text in texts:
            self.module.extract_facts(text, context)
        
        # Получаем статистику
        stats = self.module.get_stats()
        
        self.assertIn('module_stats', stats)
        self.assertIn('database_stats', stats)
        self.assertIn('cache_info', stats)
        
        module_stats = stats['module_stats']
        self.assertGreater(module_stats['total_extracted'], 0)
        self.assertIn('facts_by_type', module_stats)


class TestFactPatterns(unittest.TestCase):
    """Тесты для паттернов извлечения"""
    
    def setUp(self):
        """Инициализация"""
        from modules.extraction.fact_patterns_extended import (
            InfoUpdatePatterns,
            EnhancedFactPatterns,
            detect_question_type
        )
        
        self.info_patterns = InfoUpdatePatterns
        self.enhanced_patterns = EnhancedFactPatterns
        self.detect_question = detect_question_type
    
    def test_detect_info_update(self):
        """Тест определения обновлений информации"""
        # Тест изменения статуса
        text1 = "Я больше не работаю в банке"
        update_type, data = self.info_patterns.detect_info_update_type(text1)
        self.assertEqual(update_type, 'status_change')
        
        # Тест жизненного события
        text2 = "Вчера женился на Марине"
        update_type, data = self.info_patterns.detect_info_update_type(text2)
        self.assertEqual(update_type, 'life_event')
        self.assertEqual(data['event'], 'marriage')
        
        # Тест изменения значения
        text3 = "Раньше жил в Москве, теперь в Питере"
        update_type, data = self.info_patterns.detect_info_update_type(text3)
        self.assertEqual(update_type, 'value_change')
        self.assertIn('Москве', data.get('old_value', ''))
        self.assertIn('Питере', data.get('new_value', ''))
    
    def test_composite_facts(self):
        """Тест извлечения составных фактов"""
        # Полное имя
        text1 = "Мое полное имя Иванов Иван Иванович"
        facts = self.enhanced_patterns.extract_composite_facts(text1)
        self.assertTrue(any(f['type'] == 'full_name' for f in facts))
        
        # Полный адрес
        text2 = "Живу в Москве, улица Ленина, дом 10"
        facts = self.enhanced_patterns.extract_composite_facts(text2)
        self.assertTrue(any(f['type'] == 'full_address' for f in facts))
        
        # Образование с годом
        text3 = "Окончил МГУ в 2015 году"
        facts = self.enhanced_patterns.extract_composite_facts(text3)
        education_facts = [f for f in facts if f['type'] == 'education']
        self.assertGreater(len(education_facts), 0)
        self.assertEqual(education_facts[0]['year'], '2015')
    
    def test_detect_question_type(self):
        """Тест определения типа вопроса"""
        # info_updating
        self.assertEqual(self.detect_question("Что изменилось?"), 'info_updating')
        self.assertEqual(self.detect_question("Есть обновления?"), 'info_updating')
        
        # personal_info
        self.assertEqual(self.detect_question("Как тебя зовут?"), 'personal_info')
        self.assertEqual(self.detect_question("Сколько тебе лет?"), 'personal_info')
        
        # work_info
        self.assertEqual(self.detect_question("Где работаешь?"), 'work_info')
        self.assertEqual(self.detect_question("Какая у тебя профессия?"), 'work_info')
        
        # family_info
        self.assertEqual(self.detect_question("Ты женат?"), 'family_info')
        self.assertEqual(self.detect_question("Есть дети?"), 'family_info')
        
        # general
        self.assertEqual(self.detect_question("Как дела?"), 'general')


if __name__ == '__main__':
    unittest.main()