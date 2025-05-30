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

from openai import OpenAI

# Step 2: Define the retrieval tool (returns top 8 relevant chunks)
def document_retriever_tool(question: str, document: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(document)
    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(question)
    # Only return relevant content
    context = "\n\n".join(doc.page_content for doc in docs)
    return context

# Step 3: Convert tool to OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "document_retriever_tool",
            "description": (
                "Retrieve the most relevant parts of the document for a given question. "
                "Always use this tool to answer questions based only on what you find in the document."
            ),
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
        "You are a helpful AI assistant. For every user question, always call the document retrieval tool to search for the answer. "
        "Only answer based on the retrieved results, never by repeating or summarizing the whole document. "
        "Answer in 2-3 sentences, summarizing or quoting the most relevant evidence. "
        "Cite or quote the source chunk where possible, and answer in the same language as the user's question."
    ),
    model="gpt-4o",
    tools=tools,
)

# Step 5: Function to run agent on question + doc
def run_talk_to_data_agent(question, file_text):
    thread = client.beta.threads.create()

    # Add user message (just the question!)
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question,
    )

    # Run the assistant – no full document in the message, only as tool input
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id
    )

    # Wait for run to complete
    import time
    while run.status not in ["completed", "failed"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Get agent's answer
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    # Sometimes multiple messages, so get the last assistant reply
    latest_reply = ""
    for m in messages.data:
        for part in m.content:
            if hasattr(part, "text") and hasattr(part.text, "value"):
                latest_reply = part.text.value
    return latest_reply
