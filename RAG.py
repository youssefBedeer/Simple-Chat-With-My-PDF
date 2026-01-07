#################################
# IMPORT LIBRARIES
#################################
import os
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated

# LangChain / LangGraph
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

#################################
# LOAD ENV VARIABLES
#################################
load_dotenv()

#################################
# INITIALIZE MODELS
#################################
print("Initializing LLM + Embeddings...")

llm = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\Programming\ML\End-to-End\GenAi\HuggingFace\huggingface_models\unknown\sentence-similarity\sentence-transformers_all-MiniLM-L6-v2"
)

#################################
# LOAD PDF
#################################
pdf_path = r"D:\Programming\ML\Agentic Ai course\FreeCodeCamp course\Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

print("Loading PDF...")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"PDF loaded successfully. Total pages: {len(pages)}")

#################################
# TEXT CHUNKING
#################################
print("Chunking PDF pages...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

print(f"Text chunking complete. Total chunks: {len(chunks)}")

#################################
# VECTOR STORE
#################################
print("Building Chroma Vector Store...")

persist_directory = r"D:\Programming\ML\Agentic Ai course\FreeCodeCamp course\db"
collection_name = "stock_market"

vector_db = Chroma.from_documents(
    documents=chunks,
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding=embeddings,
)

print("Vector DB ready!")

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#################################
# LANGGRAPH SETUP
#################################
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


#################################
# TOOL: Retriever
#################################
@tool
def retriever_tool(query: str) -> str:
    """
    Search the PDF vector database and return relevant chunks.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the Stock Market Performance 2024 document."

    result = []
    for i, doc in enumerate(docs):
        result.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(result)


tools = [retriever_tool]
tools_dict = {t.name: t for t in tools}

llm = llm.bind_tools(tools=tools)

#################################
# SYSTEM PROMPT
#################################
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 using the contents of the PDF.
Use the retriever tool whenever needed. Cite relevant excerpts in your answers.
"""

#################################
# GRAPH NODES
#################################
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def take_action(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    results = []

    for call in last_msg.tool_calls:
        tool_name = call["name"]
        args = call["args"]
        query = args.get("query", "")

        if tool_name not in tools_dict:
            result = "Invalid tool name."
        else:
            result = tools_dict[tool_name].invoke(query)

        results.append(
            ToolMessage(
                tool_call_id=call["id"],
                tool_name=tool_name,
                content=str(result),
            )
        )

    return {"messages": results}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


#################################
# BUILD LANGGRAPH
#################################
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

#################################
# RUN AGENT LOOP
#################################
def running_agent():
    print("\nRAG Agent Ready — Type 'exit' to quit.\n")

    while True:
        user_input = input("\nYour question: ")

        if user_input.lower() in {"exit", "quit"}:
            print("\nExiting agent...")
            break

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===\n")
        print(result["messages"][-1].content)


running_agent()
