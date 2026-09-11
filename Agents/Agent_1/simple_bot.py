from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# load env variables
load_dotenv()

# define agent state schema
class AgentState(TypedDict):
    messages: List[HumanMessage]

# define llm
llm = ChatOpenAI(model = "gpt-4o-mini")

# define a node
def process(state: AgentState) -> AgentState:
    """This function calls the llm"""
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    return state

# define graph structure itself
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


# invoke graph agent
user_input = input("ENTER: ")
while user_input != "quit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("ENTER: ")

