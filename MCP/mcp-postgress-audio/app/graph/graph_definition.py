from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import List

from dotenv import load_dotenv

#### Importing local graph components
from .system_prompt import system_message
from .graph_state import AgentState

#### Importing local database definitions
from ..db_definitions.pydantic_models import ExpenseCategory


load_dotenv()

# import logging
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)


class Agent:
    def __init__(
            self,
            model: str = "gpt-4o-mini",
            tools: List[BaseTool] = [],
            system_prompt: str = system_message,
            ) -> None:        
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools

        self.llm = ChatOpenAI(model=model).bind_tools(tools=self.tools)
        self.graph = self.build_graph()

    def build_graph(self,) -> CompiledStateGraph:
        builder = StateGraph(AgentState)

        builder.add_node(self.assistant)
        builder.add_node(ToolNode(self.tools))

        builder.set_entry_point("assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition
        )
        builder.add_edge("tools", "assistant")

        return builder.compile(checkpointer=InMemorySaver())
    
    def assistant(self, state: AgentState):
        """The main assistant node that uses the LLM to generate responses."""
        # inject customer_id from the state into the system prompt
        system_prompt = self.system_prompt.format(
            customer_id=state.customer_id,
            expense_categories=", ".join([c.value for c in ExpenseCategory])
            )

        response = self.llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
        state.messages.append(response)
        return state

    def draw_graph(self,):
        if self.graph is None:
            raise ValueError("Graph not built yet")
        from IPython.display import Image

        return Image(self.graph.get_graph().draw_mermaid_png())

# Initial testing
agent = Agent()
state = AgentState(messages=[], customer_id="12345")
print(agent.assistant(state))

if __name__ == "__main__":
    agent.draw_graph()
