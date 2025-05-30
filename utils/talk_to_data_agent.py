# utils/talk_to_data_agent.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

# Step 1: Safely get the OpenAI API key
try:
    import streamlit as st
    api_key = st.secrets["OPENAI_API_KEY"]
except (ModuleNotFoundError, KeyError):
    api_key = os.getenv("OPENAI_API_KEY")

import openai
from openai import OpenAI

# Step 2: Define your tool
def document_retriever_tool(question: str, document: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(document)
    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 12})

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

# Step 3: Convert it to an OpenAI tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "document_retriever_tool",
            "description": "Retrieve the most relevant parts of the document for a given question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "document": {"type": "string"},
                },
                "required": ["question", "document"],
            },
        },
    }
]

# Step 4: Create the agent (assistant)
client = OpenAI(api_key=api_key)

agent = client.beta.assistants.create(
    name="Talk to Your Data Agent",
    instructions=(
        "You are a helpful AI assistant. Always use the document retrieval tool to search the document and answer questions directly using content from the provided document. "
        "If the user has uploaded a document, never ask for more documents. Instead, do your best to answer based only on what is in the provided content. Quote or reference the document where possible. "
        "When answering, do not repeat or paste the whole document. Always answer in 2-3 sentences, summarizing or quoting only the most relevant portion. If possible, cite exactly where you found the information. "
        "Answer in the same language as the user's question if possible."
    ),
    model="gpt-4o",
    tools=tools,
)

# Step 5: Function to run agent on question + doc
def run_talk_to_data_agent(question, file_text):
    # Debug print: Show what's being sent
    print("=== Document content being sent to agent (first 1000 chars) ===")
    print(file_text[:1000])  # Only print first 1000 characters
    print("=== User question being sent to agent ===")
    print(question)

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Here is the document for analysis:\n\n{file_text}\n\nMy question is: {question}",
    )
    # ...rest of code...

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id
    )

    # Wait for run to complete
    import time
    while run.status not in ["completed", "failed"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    latest_reply = messages.data[0].content[0].text.value
    return latest_reply
