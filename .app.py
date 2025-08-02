from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv                                                                                                      # type: ignore
from langchain_core.messages import BaseMessage                                                                                                     # type: ignore
from langchain_core.messages import ToolMessage                                                                                                     # type: ignore
from langchain_core.messages import SystemMessage                                                                                                   # type: ignore
from langchain_openai import ChatOpenAI                                                                                                 # type: ignore
from langchain_core.tools import tool                                                                                                   # type: ignore
from langgraph.graph.message import add_messages                                                                                                    # type: ignore
from langgraph.graph import StateGraph, END                                                                                                 # type: ignore
from langgraph.prebuilt import ToolNode                                                                                                 # type: ignore


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)


def model_call(state:AgentState) -> AgentState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability."), ("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))