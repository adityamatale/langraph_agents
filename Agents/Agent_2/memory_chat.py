from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# load env variables
load_dotenv()

# define agent state schema
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# define llm
llm = ChatOpenAI(model="gpt-4o-mini")

# define node
def process(state: AgentState) -> AgentState:
    """This function processes the user input"""

    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print(f"\nCURRENT STATE: {state["messages"]}\n")

    return state


# build graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


#invoke agent
conversation_history = []

user_input = input("ENTER: ")
while user_input.lower() != "quit":
    # append user input to conversation history
    conversation_history.append(HumanMessage(content=user_input))

    # invoke agent
    result = agent.invoke({"messages": conversation_history})

    # print(result['messages'])
    conversation_history = result['messages']

    # get next input
    user_input = input("ENTER: ")


with open ("logging.txt", "w") as file:
    file.write("Conversation Logs:\n")
    for messages in conversation_history:
        if isinstance(messages, HumanMessage):
            file.write(f"User: {messages.content}\n")
        elif isinstance(messages, AIMessage):
            file.write(f"AI: {messages.content}\n")
    file.write("\nEnd of Conversation\n")

print("Conversation logged to logging.txt")