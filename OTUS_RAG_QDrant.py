import os
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Инструменты RAG
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
#from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Импорты LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

# Импорты LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Визуализация
import pandas as pd
from IPython.display import display, Markdown

# Конфиг
DATA_DIR = Path("../data/")
COLLECTION_NAME = "client_docs"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-m3"

import torch
#from pre_data import parse_faq_text,read_text_file 
from convert_data_for_chanck import load_and_enrich_documents

from langchain_community.embeddings import OllamaEmbeddings

# настройка для работы судьи
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)

if __name__ == "__main__":

# проверка устройства 
    if torch.backends.mps.is_available():
        device = "mps"   # GPU на Mac (Apple Silicon)
    elif torch.cuda.is_available():
        device = "cuda"  # GPU NVIDIA (если вдруг есть)
    else:
        device = "cpu"

    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("   GPU: Apple Silicon (M‑series) via MPS")   

# подключение landfuse для мониторинга 
    try:
            langfuse_handler = CallbackHandler()
            print("✅ Langfuse мониторинг подключен")
    except Exception as e:
            print(f"⚠️ Ошибка подключения Langfuse: {e}")
            langfuse_handler = None

# Добавляем его в конфиг
    callbacks: list[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []
    config: RunnableConfig = {
            "configurable": {"thread_id": "session_1"},
            "callbacks": callbacks,
    }

# проверка  моделм эмбедингов 
    print(f"⏳ подключение к модели эмбеддингов {EMBEDDING_MODEL} на {device}...")
    test_vec = embeddings.embed_query("Тестовый запрос")
    print(len(test_vec))

# подключение к векторной базе развернутой в Docker
    client = QdrantClient(url=QDRANT_URL)

    # Проверяем, есть ли коллекция, и пересоздаем её для чистоты эксперимента
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Удалена старая коллекция {COLLECTION_NAME}")

    print(f"🛠 Создание коллекции с HNSW индексом...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        # размер вектора 1024 определяется размером вектора embadding модели
        # используем Distance.COSINE оптимально для текста, когда важен  угол а не абсолютное значение вектора 
        # как при параметре Distance.EUCLID
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        # Настройка HNSW 
        hnsw_config=models.HnswConfigDiff(
            m=16,               # Количество связей на узел (больше = точнее, но больше памяти)
            ef_construct=100    # Глубина поиска при построении индекса
        )
    )
    print("✅ Коллекция готова!")
        
    print("⏳ Запуск индексации документов (может занять время)...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    file_path = "data/Шаблоны_ответов_ для_ИИ.txt"
    docs=load_and_enrich_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,      # Указываем размер с запасом, фактически текст будет меньше и разделять символами разделения
        chunk_overlap=200,    # Нахлест, чтобы не терять контекст на границах
        separators=["\n\n","\n## ",  "\n", " ", ""] # Приоритет разделителей (два перевода строки)
    )

    splits = text_splitter.split_documents(docs)
    
    print(f"📚 Исходных документов: {len(docs)}")
    print(f"✂️  Получено чанков: {len(splits)}")

# вывод на печать, получившихся чанков для тестирования 
    #for entry in splits:
    #    print(f"Вопросы: {entry.page_content}")      # ✅
    #    print(f"Категория: {entry.metadata['category']}")      # ✅
    #    print(f"Субкатегория: {entry.metadata['sub_category']}") # ✅
    #    print(f"Ответ: {entry.metadata['Answer']}")      # ✅
    #    print("-" * 50)
        

#   Добавление в векторную базу чанков документов
    vector_store.add_documents(splits)
    print(f"🎉 Успешно проиндексировано {len(splits)} чанков.")

# собираем цепочку 
    # 1. Инициализация LLM
    llm = ChatOllama(
        model="mistral:7b", 
        temperature=0.1, # Низкая температура для фактологической точности
        base_url="http://localhost:11434",
        callbacks=callbacks  # <-- добавляем callback
    )

    # 2. Промпт (Инструкция)
    # Используем шаблон, который заставляет модель опираться ТОЛЬКО на контекст.
    template = """Ты — корпоративный ассистент компании ЭнергосбыТ Плюс".
    Ответь на вопрос клиента, используя ТОЛЬКО предоставленный ниже контекст.
    Если в контексте нет информации, скажи "В документах нет информации об этом".
    Не придумывай факты.

    Контекст:
    {context}

    Вопрос: {question}

    Ответ:"""


    prompt = ChatPromptTemplate.from_template(template)

    # 3. Функция форматирования документов в строку

    def format_docs_score(docs_and_scores):
        # docs_and_scores — это список вида [(doc, score), ...]
        formatted_parts = []
        for i, (doc, score) in enumerate(docs_and_scores):
            # Получаем текст ответа из metadata
            answer = doc.metadata.get("Answer", "")
            # Получаем page_content
            content = doc.page_content
            # Формируем строку для текущего документа
            formatted_part = f"[Документ {i+1}] Score: {score:.4f}\n[Answer: {answer}]\n{content}"
            formatted_parts.append(formatted_part)

        # Соединяем все части, разделяя двумя переносами строки
        return "\n\n".join(formatted_parts)



    # 4. Ретривер использует метод as_retriever (не возвращает оценку Score)  

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.66  # добавляем порог схожести для отсеивания отобранных докментов 
                    }
    )

    # 4.1 Ретривер использует метод imilarity_search_with_score ( возвращает оценку Score)  
    def custom_retrieve(query: str):
        results = vector_store.similarity_search_with_score(query, k=3)
        return results

    retriever_lambda = RunnableLambda(custom_retrieve)

    # 5. вывод результата поиска (переменная context)на экран 
    def print_context(inputs):
        print("Context:\n", inputs["context"])  # выводим context
        return inputs  # возвращаем тот же словарь
    
    # 5.1 вывод результата поиска c оценкой + переменная context на экран 
    def print_context_with_scores(inputs):
        context_docs_and_scores = inputs.get("context_docs_and_scores", [])
        print("\n--- Документы и их оценки (scores) ---")
        for i, (doc, score) in enumerate(context_docs_and_scores):
            Answer = doc.metadata.get('Answer', 'N/A')
            category = doc.metadata.get('category', 'N/A')
            print(f"[{i+1}] Score: {score:.4f} | Ответ: {Answer} | Категория: {category}")
            print(f"    Текст: {doc.page_content[:100]}...\n")

        print("Context (форматированный):\n", inputs["context"])
        return inputs
    
    # 6. Сборка цепочки (LCEL - LangChain Expression Language)
    rag_chain = (
            {
                "context_docs_and_scores": retriever_lambda,
                "context": retriever_lambda | format_docs_score,
                "question": RunnablePassthrough()
            }
         | RunnableLambda(print_context_with_scores)
         | prompt
         | llm
         | StrOutputParser()
    )

    print("🔗 RAG Chain собрана!")


    question = "Я купил дом с установленными счетчиками. Какие документы мне нужно подготовить для оформления договора на коммунальные услуги?"
    print(f"❓ Вопрос: {question}\n")
    print("⏳ Генерация ответа (Mistral думает)...\n")
    # Запуск
    response = rag_chain.invoke(question)
    print("🤖 Ответ ассистента:")
    print(response)

# настройка цепочки судьи -------------------------------------------------------------------------------------
    # 1. Определяем структуру ответа Судьи (через Pydantic)
    class Grade(BaseModel):
        score: int = Field(description="Оценка от 1 до 5, где 5 - идеальный ответ")
        is_faithful: bool = Field(description="True, если ответ не содержит галлюцинаций и основан на контексте")
        reasoning: str = Field(description="Краткое объяснение оценки")

    # 2. Парсер для преобразования ответа LLM в JSON
    parser = JsonOutputParser(pydantic_object=Grade)

    # 3. Промпт для Судьи
    judge_template = """Ты — беспристрастный судья, оценивающий качество RAG-системы.
    Проанализируй Входные данные и оцени качество Ответа Ассистента.

    Входные данные:
    1. ВОПРОС пользователя: {question}
    2. КОНТЕКСТ (найденные документы): {context}
    3. ОТВЕТ Ассистента: {answer}

    Критерии оценки:
    - Faithfulness: Ответ должен опираться ТОЛЬКО на предоставленный контекст. Если Ассистент добавил информацию "от себя" — это плохо.
    - Relevance: Ответ должен четко отвечать на вопрос пользователя.

    Формат вывода (JSON):
    {format_instructions}
    """

    judge_prompt = ChatPromptTemplate.from_template(
        template=judge_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 4. Цепочка оценки
    judge_chain = judge_prompt | llm | parser

    print("⚖️ Судья готов к работе.")

# запуск судьи для оценки ответа 

# Превратим список документов обратно в строку для подачи Судье
    docs_and_scores = retriever_lambda.invoke(question)
    context_string = format_docs_score(docs_and_scores)

    evaluation = judge_chain.invoke({
        "question": question,
        "context": context_string,
        "answer": response  
    })

    print("📊 Вердикт Судьи:")
    print(evaluation)
    
    #print(f"Оценка: {evaluation['score']}/5")         # ✅ Обращение к словарю
    #print(f"Достоверность (Faithful): {evaluation['is_faithful']}")
    #print(f"Причина: {evaluation['reasoning']}")
    
    print(f"Оценка:", evaluation.score)  
    print(f"Достоверность:",evaluation.is_faithful)   
    print(f"Причина:", evaluation.reasoning)   

# проверка на галюцинации

    fake_question = "Почем яблоки в соседнем киоске?"
    print(f"❓ Вопрос: {fake_question}")
    print(f"⚖️ Оценка ответа на вопрос: '{fake_question}'...\n")
    response_fake = rag_chain.invoke(
         {fake_question},
         config={"callbacks": [langfuse_handler]}
   )
    print("\n🤖 Ответ ассистента:")
    print(response_fake)