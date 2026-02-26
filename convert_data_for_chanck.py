import re
from typing import List
from langchain_core.documents import Document  # Изменили импорт


def load_and_enrich_documents(file_path: str) -> List[Document]:
    """
    Загружает один текстовый файл, разбивает его на блоки по строке,
    начинающейся с '# ' (например, '# Все', '# Блок 1', и т.д.),
    извлекает метаданные и формирует список документов LangChain.

    1) Функция на вход получает текстовый файл.
    2) Разбивает содержимое текста на блоки ключом разделителем является символ # в начале строки.
    3) В каждом блоке первая строка (после удаления '# ') записывается в doc.metadata["category"].
    4) Следующая строка после символов '## ' записывается в doc.metadata["sub_category"].
    5) Последующие строки после символов '## ' записываются через запятую в doc.page_content.
    6) Текст после '**Ответ: **' записывается в doc.metadata["Answer"].
    """
    # 1. Чтение содержимого файла
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Разделение текста на блоки по строке, начинающейся с '# '
    # Используем re.split, ищем строки вида "^# " (начало строки + # + пробел)
    # flags=re.MULTILINE позволяет ^ соответствовать началу строки
    # Мы не учитываем слово "Все", только символ #
    blocks = re.split(r'^#\s.*$', content, flags=re.MULTILINE)

    # Теперь нужно получить сами заголовки '# ...' для связывания с блоками
    # Используем findall, чтобы получить все строки, начинающиеся с '# '
    headers = re.findall(r'^#\s.*$', content, flags=re.MULTILINE)

    # Удаляем начальный и конечный пробелы/пустые строки из блоков
    processed_blocks = [block.strip() for block in blocks if block.strip()]

    enriched_docs = []

    # Проходим по каждому блоку и соответствующему ему заголовку
    for idx, block_text in enumerate(processed_blocks):
        # Извлекаем соответствующий заголовок
        category = headers[idx].strip('# ').strip() if idx < len(headers) else "Неизвестная категория"

        lines = block_text.splitlines()
        # Удаляем пустые строки и лишние пробелы
        lines = [line.strip() for line in lines if line.strip()]

        # 4. metadata["sub_category"] - следующая строка после '## '
        sub_category = ""
        content_lines = []
        answer_lines = []
        collecting_answer = False

        for line in lines:
            if line.startswith('## ') and not sub_category:
                # 4. Первая строка после ## -> sub_category
                sub_category = line.strip('## ').strip()
            elif line.startswith('## ') and sub_category:
                # 5. Последующие строки после ## -> в content_lines
                content_lines.append(line.strip('## ').strip())
            elif line.startswith('**Ответ: **'):
                # 6. Начинается текст после '**Ответ: **'
                collecting_answer = True
                # Берём остаток строки после '**Ответ: **'
                answer_part = line[len('**Ответ: **'):].strip()
                if answer_part:
                    answer_lines.append(answer_part)
            elif collecting_answer:
                # 6. Продолжение текста ответа
                answer_lines.append(line)
            elif not collecting_answer and not line.startswith('## '):
                # Если строка не является ## и ответ ещё не начат, добавляем в content
                # (может быть случай, когда после category сразу идёт текст до ##)
                # В текущем формате данных, всё равно идёт ## первой, но на всякий случай
                # включим эту строку в content, если она не относится к ответу
                # Однако, если строка уже после ##, она обработана выше.
                # Т.е. строки между ## и **Ответ:** идут в content_lines
                # и они уже добавлены.
                pass # Все строки до **Ответ:** уже обработаны выше

        # 5. page_content: последующие строки после '## ', через запятую
        page_content_str = ', '.join(content_lines).strip()

        # 6. metadata["Answer"]: текст после '**Ответ: **'
        answer_str = '\n'.join(answer_lines).strip()

        # Создаем документ LangChain
        doc = Document(
            page_content=page_content_str,
            metadata={
                "category": category,       # 3
                "sub_category": sub_category, # 4
                "Answer": answer_str,       # 6
                "source": file_path         # Источник файла
            }
        )
        enriched_docs.append(doc)

    return enriched_docs

# --- использование ---
if __name__ == "__main__":
    #  путь к вашему текстовому файлу
    file_path = "data/Шаблоны_ответов_ для_ИИ.txt"  

    docs = load_and_enrich_documents(file_path)

    for i, doc in enumerate(docs):
        print(f"\n--- Документ {i+1} ---")
        print(f"Page Content: {repr(doc.page_content)}")
        print(f"Metadata: {doc.metadata}")