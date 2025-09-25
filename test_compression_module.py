#!/usr/bin/env python3
"""
Тест Compression Module
Проверяем все методы интерфейса ICompressor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from submit.modules.compression.module import CompressionModule

def test_compression_module():
    """Тестируем Compression Module"""
    print("🧪 Тестируем Compression Module...")
    
    # Конфигурация
    config = {
        'min_confidence': 0.5,
        'use_rules': True,
        'use_llm': False,
        'conflict_strategy': 'latest',
        'filter_copypaste': True,
        'max_text_length': 10000,
        'rule_confidence_threshold': 0.7,
        'method': 'hybrid'
    }
    
    try:
        # Создаем модуль
        compressor = CompressionModule(config)
        print("✅ CompressionModule создан успешно")
        
        # Тест 1: compress_text
        print("\n📝 Тест 1: compress_text")
        test_text = "Это очень длинный текст для тестирования сжатия. Он содержит много информации, которая может быть сжата без потери смысла. Мы проверяем, как работает наш компрессор на различных уровнях сжатия."
        
        result = compressor.compress_text(test_text, "moderate")
        assert result.success, f"compress_text failed: {result.error}"
        assert result.data is not None, "compress_text returned None"
        assert len(result.data) < len(test_text), "Сжатый текст должен быть короче оригинального"
        print(f"✅ compress_text: {len(test_text)} -> {len(result.data)} символов")
        
        # Тест 2: compress_sessions
        print("\n📝 Тест 2: compress_sessions")
        sessions = {
            'session_1': "Пользователь рассказал о своей работе программистом в компании Google.",
            'session_2': "Обсуждение хобби - игра в футбол по выходным с друзьями.",
            'session_3': "Планы на отпуск - поездка в Италию в следующем месяце."
        }
        
        result = compressor.compress_sessions(sessions)
        assert result.success, f"compress_sessions failed: {result.error}"
        assert result.data is not None, "compress_sessions returned None"
        assert len(result.data) == len(sessions), "Количество сжатых сессий должно совпадать"
        print(f"✅ compress_sessions: {len(sessions)} сессий сжато")
        
        # Тест 3: get_compression_stats
        print("\n📝 Тест 3: get_compression_stats")
        result = compressor.get_compression_stats()
        assert result.success, f"get_compression_stats failed: {result.error}"
        assert result.data is not None, "get_compression_stats returned None"
        assert 'total_compressed' in result.data, "Статистика должна содержать total_compressed"
        print(f"✅ get_compression_stats: {result.data}")
        
        # Тест 4: compress_for_context (дополнительный метод)
        print("\n📝 Тест 4: compress_for_context")
        long_text = "Это очень длинный текст, который нужно сжать до определенной длины. " * 10
        compressed = compressor.compress_for_context(long_text, 100)
        assert len(compressed) <= 100, f"Сжатый текст слишком длинный: {len(compressed)}"
        print(f"✅ compress_for_context: {len(long_text)} -> {len(compressed)} символов")
        
        # Тест 5: Различные уровни сжатия
        print("\n📝 Тест 5: Различные уровни сжатия")
        for level in ["light", "moderate", "heavy"]:
            result = compressor.compress_text(test_text, level)
            assert result.success, f"compress_text with level {level} failed: {result.error}"
            print(f"✅ Уровень {level}: {len(test_text)} -> {len(result.data)} символов")
        
        print("\n🎉 Все тесты Compression Module прошли успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте Compression Module: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_compression_module()
    sys.exit(0 if success else 1)
