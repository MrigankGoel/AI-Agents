from typing import TypedDict, List
from langchain_core.messages import HumanMessage                                                                                                                                                # type: ignore
from langchain_openai import ChatOpenAI                                                                                                                                             # type: ignore
from langgraph.graph import StateGraph, START, END                                                                                                                                              # type: ignore
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values                                                                                                                                               # type: ignore

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
