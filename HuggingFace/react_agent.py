from smolagents import CodeAgent, ToolCallingAgent
from langchain import hub
from langchain.tools.render import render_text_description
from tools import Tools
from llm import LLM
from dotenv import load_dotenv

load_dotenv()
class ReactAgent:
    def __init__(self):
        self.prompt = hub.pull("hwchase17/react-json")
        self.tools_obj = Tools()
        self.tools = self.tools_obj.get_tools()
        self.llm = LLM().llm
        self.prompt = self.prompt.partial(
            tools=render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        self.tools_agent = ToolCallingAgent(
            tools=self.tools,
            model=self.llm,
            name="ToolCallingAgent",
            description="Manages tool execution and web searches; it DOES NOT generate Python code to execute web searches. It also manages final state of responses."
        )

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.llm,
            managed_agents = [self.tools_agent],
            additional_authorized_imports=['time', 'requests', 'pandas', 'numpy'],
            planning_interval=2
        )
    def run(self, query:str):
        return self.agent.run(query)
    
    def get_memory(self):
        return self.agent.memory.get_full_steps()
    