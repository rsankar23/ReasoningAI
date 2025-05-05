from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools import WikiTool

class ReActAgent:
    def __init__(self, llm, prompt, tools):
        self.llm = llm
        self.prompt = prompt
        self.tools = tools
        self.agent_core = create_react_agent(
            llm = self.llm,
            tools = self.tools,
            prompt = self.prompt
        )
        self.executor = AgentExecutor(
            agent = self.agent_core,
            return_intermediate_steps = True,
            tools = self.tools,
            verbose = True
        )
