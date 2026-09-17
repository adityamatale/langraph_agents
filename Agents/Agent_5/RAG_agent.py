from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

# llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # more deterministic output

# embedding model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small',)

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

pdf_Loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_Loader.load()
    print(f"Loaded {len(pages)} pages from PDF.")
except Exception as e: 
    print(f'Error loading PDF: {e}')
    raise

# chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_splitter.split_documents(pages)


# vector db - local
persist_dir = r'/home/matty/Documents/UE_DataScience/Sem_II/generativeAI_for_dataScience/langgraph_beginner/Agents/Agent_5/chroma_db/'

collection_name = "stock_market_perform"

# make dir if not exists
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# create the vector db
try:
    # create db using the embedding model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print("Vector DB created successfully.")

except Exception as e:
    print(f"Error creating Vector DB: {e}")
    raise


# now we create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# create a retriever tool

@tool
def retriever_tool(query: str)-> str:
    """Use this tool to retrieve information about the stock market performance from the pdf"""

    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevent in the stock market performance 2024 document"


    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

# create agent state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# conditional edge function
def should_continue(state:AgentState)-> AgentState:
    """Check if the last message contains tool calls"""
    result = state['messages'][-1]

    return hasattr(result, "tool_calls") and len(result.tool_calls)>0


system_prompt = SystemMessage(content="""
    You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
    Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers.
""")

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# llm agent
def call_llm(state: AgentState) -> AgentState:
    """Node to call the LLM with the current state"""
    messages = list(state['messages'])
    # ensure system prompt is included
    messages = [system_prompt] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}



# retriever agent
def take_action(state:AgentState)-> AgentState:
    """Node to take action or execute tool calls from llm response"""
    tool_calls = state['messages'][-1].tool_calls

    results=[]
    for t in tool_calls:
        print(f"Calling tool: {t['name']} with query: {t['args'].get('query', 'No query found')}")

        if not t['name'] in tools_dict:
            print(f'\n Tool: {t['name']} does not exist')
            result = 'Incorrect tool name , please retry and select tools from list of available tools.'

        else:
            result = tools_dict[t['name']].invoke(t['args'].get("query", ""))
            print(f'result length: {len(str(result))}')

        # append tool message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print(f'Tool execution comlete. back to model')

    return {'messages': results}



# create graph
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_edge("retriever_agent", "llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("======RAG AGENT======")

    while True:
        user_input = input("USER: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        messages = [HumanMessage(content=user_input)] 

        result = rag_agent.invoke({"messages": messages})

        print(f"======ANSWER======")
        print(result['messages'][-1].content)


running_agent()