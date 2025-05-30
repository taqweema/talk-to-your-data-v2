# talk_to_data_agent.py
import os
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

# Step 1: Define your tool
def document_retriever_tool(question: str, document: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(document)

    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_texts(chunks, embedding=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(doc.page_content for doc in docs)
    return context

# Step 2: Convert it to OpenAI tool
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

# Step 3: Create the agent
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

agent = client.beta.assistants.create(
    name="Talk to Your Data Agent",
    instructions="""You are a helpful AI assistant. Use tools to retrieve, understand, and answer questions from long documents. Always explain your reasoning and cite sources where possible.""",
    model="gpt-4o",
    tools=tools,
)

# Step 4: Function to run agent on question + doc
def run_talk_to_data_agent(question, file_text):
    thread = client.beta.threads.create()

    # Add user message
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"My question is: {question}",
        file_attachments=[],  # Optional: for future use
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent.id,
        tool_choice="auto",
        additional_instructions="Here is the full document text you can use via your tools.",
        tool_inputs={
            "document_retriever_tool": {
                "question": question,
                "document": file_text,
            }
        }
    )

    # Wait for run to complete
    while run.status not in ["completed", "failed"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    latest_reply = messages.data[0].content[0].text.value
    return latest_reply
