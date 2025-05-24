import os
import re
import openai
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def extract_pdf_pages_with_text(text_chunks, pdf_file_path):
    reader = PdfReader(pdf_file_path)
    page_sources = []
    for chunk in text_chunks:
        found = False
        for i, page in enumerate(reader.pages):
            if chunk[:100].strip() in page.extract_text():
                page_sources.append((chunk, i + 1))  # page number starts at 1
                found = True
                break
        if not found:
            page_sources.append((chunk, None))
    return page_sources


def process_and_query(document_text, user_question, return_sources=False):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(document_text)

    # Step 2: Embed chunks
    embeddings = OpenAIEmbeddings()

    # Step 3: Use temporary vector DB
    with tempfile.TemporaryDirectory() as tmpdir:
        # Disable LangChain & Chroma telemetry (fixes protobuf errors in deployment)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_API_KEY"] = ""
        os.environ["LANGCHAIN_PROJECT"] = ""
        os.environ["LANGCHAIN_ENDPOINT"] = ""
        os.environ["LANGCHAIN_CLIENT"] = "false"

        vectordb = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            persist_directory=tmpdir
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 6})
        docs = retriever.get_relevant_documents(user_question)

    # Step 4: Generate context and reference map
    context = []
    source_map = {}
    page_refs = {}
    for i, doc in enumerate(docs):
        ref_number = i + 1
        context.append(f"[{ref_number}] {doc.page_content}")
        source_map[ref_number] = doc.metadata
        page_refs[ref_number] = doc.metadata.get("page", "?")

    combined_context = "\n\n".join(context)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that uses citation numbers like [1], [2] in answers."},
        {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {user_question}"}
    ]

    # Step 5: Call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3
    )
    answer = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})

    if return_sources:
        refs = list(set(int(num) for num in re.findall(r"\[(\d+)\]", answer)))
        source_details = {}
        for ref in refs:
            meta = source_map.get(ref, {})
            source_details[ref] = {
                "page_number": page_refs.get(ref, "?"),
                "excerpt": context[ref - 1][5:300] + "..."  # short preview
            }

        return {
            "answer": answer,
            "usage": usage,
            "sources": source_details
        }

    return {"answer": answer, "usage": usage}
