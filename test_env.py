import sys
print(f"Python interpreter: {sys.executable}")
try:
    import langchain
    import qdrant_client
    import ollama
    print("✅ Все зависимости доступны")
except ImportError as e:
    print(f"❌ Ошибка: {e}")