"""Module responsible for creating LangGraph agent and invoking it to test results"""
import os
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage           # type: ignore
from langgraph.graph import StateGraph, END                                            # type: ignore
from langgraph.graph.message import add_messages                                       # type: ignore
from langchain_groq import ChatGroq                                                    # type: ignore
import tool_def as tldf

# Load LLM model and bind tools
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llama3 = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-70b-versatile") # llama-3.1-405b-reasoning

class AgentState(TypedDict):
    """Defining state of nodes to be used for LAM graph"""
    messages: Annotated[list, add_messages]

class YfAgent:
    """Class object for custom YF Agent"""
    def __init__(self, model, tools, system_msg=""):
        self.system = system_msg
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tldf.tools, tool_choice="auto")

        graph = StateGraph(AgentState)
        graph.add_node("model_inference", self.call_llm) # (<node_name>, <agent/func_name>)
        graph.add_node("action", self.use_tool_actions)
        graph.add_conditional_edges(
            "model_inference",
            self.does_tool_exist,
            {True: "action", False: END}
        )
        graph.add_edge("action", "model_inference") # (<node_name1>, <node_name2>)
        graph.set_entry_point("model_inference")
        self.graph = graph.compile()

    def does_tool_exist(self, state: AgentState) -> bool:
        """Checks whether relevant tool exists or not."""
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def call_llm(self, state: AgentState):
        """Invokes LLM with custom prompt if any."""
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        ai_message = self.model.invoke(messages)
        return {'messages': ai_message}
    
    def use_tool_actions(self, state: AgentState):
        """Uses existing tools and appends to message state for model to interpret."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in self.tools:
                print(f"Tool with name: {t['name']} not found in arsenal.")
                result = "Alas! Tool not found. Please try again with different tools."
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            return {'messages': results}

       
prompt="""
You are a smart financial analyst capable of retrieving information about companies.
Use the list of tools at your disposal to answer specific questions.
Take your time and look up information before answering the question if needed.
"""

# Define custom RAG agent
rag_agent = YfAgent(llama3, tldf.tools, system_msg=prompt)

# Test Results
messages = [HumanMessage(content="What is EBITDA of apple and google?")]
result = rag_agent.graph.invoke({"messages": messages})
print(result['messages'][-1].content)