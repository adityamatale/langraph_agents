from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # functional class for all message types in langgraph
from langchain_core.messages import ToolMessage # passes data back to llm after it calls a tool
from langchain_core.messages import SystemMessage # message for providing instructions to the llm
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()


# Annotated: type annotation which provides additional context without affecting the type itself | like meta data
"""
email = Annotated[str, "this is meta data for email"]
print(email.__metadata__)
"""
# Sequence: to automatically handle state updates for sequence such as by adding new messages to a chat history

# add_messages: its a reducer function - its a rule that controls how updates from nodes are combined with the existing state | tells us how to merge new data into state | 
#                                       | without reducer, updates will simply replace the existing values.

# create state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # annotated[datatype, metadata]


# create first tool
@tool
def add(a: int, b: int):
    """Addition function to add two numbers together"""
    return a + b

@tool
def multiply(a: int, b: int):
    """Multiplication function to multiply two numbers together"""
    return a * b

@tool
def subtract(a: int, b: int):
    """Subtraction function to subtract two numbers together"""
    return a - b

@tool
def divide(a: int, b: int):
    """Division function to divide two numbers together"""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

# list of tools
tools = [add, multiply, subtract, divide]


# define llm model
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# define a node
def model_call(state: AgentState) -> AgentState:
    """This is a node function that calls tools"""

    system_prompt = SystemMessage(content="You are my AI assistant, Please answer my queries to the best of your abilities.")

    response = model.invoke([system_prompt]+state["messages"]) # system prompt + user input

    print(f"\nCURRENT STATE: {state["messages"]}\n")

    return {"messages": [response]}     # reducer function appends the msg here instead of overriding it 


# conditional edge for looping logic
def should_continue(state: AgentState) -> AgentState:
    """This is a conditional edge where it decides to loop or not"""

    messages = state['messages']
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end" # edges to be defined later in the graph
    else:
        return "continue" # edges to be defined later in the graph


# create the actual graph structure
graph = StateGraph(AgentState)

graph.add_node("my_agent", model_call) # (name, function/action )

# create tool node
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("my_agent") # defines where to start
graph.add_conditional_edges(
    "my_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "my_agent") # loops back to agent node

app = graph.compile()

# helper function for outputing the state
def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print() 

# input = {'messages': [('user', 'add 34+21. and then substract 3-2. multiply 12*12. and finally divide 4/2')]}
input = {'messages': [('user', 'add 34+21  and then multiply the result by 2. also tell me a dad joke after that.')]}


print_stream(app.stream(input, stream_mode='values'))