# Команды для удаления устаревших файлов
# ВНИМАНИЕ: Сначала создайте бэкап!

# 1. Создание бэкапа
cp -r src/submit src/submit_backup

# 2. Удаление устаревших версий model_inference
rm src/submit/model_inference_original.py
rm src/submit/model_inference_v2.py
rm src/submit/model_inference_vector.py

# 3. Удаление устаревших компонентов RAG
rm src/submit/rag/vector_rag_engine.py
rm src/submit/rag/vector_rag_interface.py

# 4. Удаление устаревших компонентов embeddings
rm src/submit/embeddings/embedding_engine.py
rm src/submit/embeddings/vector_store.py
rm src/submit/embeddings/vector_models.py
rm src/submit/embeddings/vector_utils.py
rm src/submit/embeddings/test_vector_search.py

# 5. Удаление неиспользуемых компонентов
rm src/submit/questions/classifier.py
rm src/submit/questions/confidence.py
rm src/submit/questions/topics.py
rm src/submit/ranking/scorer.py
rm src/submit/ranking/session_ranker.py
rm src/submit/prompts/fallback_prompts.py
rm src/submit/prompts/topic_prompts.py

# 6. Удаление устаревших конфигураций
rm src/submit/config_loader.py
rm src/submit/config.yaml

# 7. Удаление неиспользуемых компонентов core
rm src/submit/core/data_loader.py
rm src/submit/core/message_filter.py

# 8. Удаление пустых папок (если они пустые)
rmdir src/submit/questions 2>/dev/null || true
rmdir src/submit/ranking 2>/dev/null || true
rmdir src/submit/prompts 2>/dev/null || true

# 9. Очистка кэша Python
find src/submit -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find src/submit -name '*.pyc' -delete 2>/dev/null || true

echo 'Очистка завершена! Проверьте систему.'