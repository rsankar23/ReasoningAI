from smolagents import CodeAgent, ToolCallingAgent
from langchain import hub
from langchain.tools.render import render_text_description
from tools import Tools
from llm import LLM
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from rich import print
load_dotenv()


class ReasoningAgent:
    def __init__(self):
        self.prompt = hub.pull('langchain-ai/react-agent-template').partial(
            tools = render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        self.tools_obj = Tools()
        self.tools = self.tools_obj.get_tools()
        self.llm = LLM().get_llm()
        self.tools_agent = ToolCallingAgent(
                tools=self.tools,
                model=self.llm,
                max_steps = 10,
                name="ToolCallingAgent",
                description="Manages tool execution and web searches; it DOES NOT generate Python code to execute web searches. It also manages final state of responses."
            )

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.llm,
            managed_agents = [self.tools_agent],
            additional_authorized_imports=['time', 'requests', 'pandas', 'numpy'],
            # planning_interval=10
        )
    def run(self, query:str):
        return self.agent.run(query)
    
    def set_planning_interval(self, interval:int):
        try:
            self.agent.planning_interval = interval
            print(f"[green] Interval Set: {interval}")
        except Exception as e:
            print(f"[red] Error on setting interval: {e}")

    def set_max_planning_steps(self, steps:int):
        try:
            self.tools_agent.max_steps = steps
            print(f"[green] Max Steps: {steps}")
        except Exception as e:
            print(f"[red] Error on setting max steps: {e}")

    
    def get_memory(self):
        return self.agent.memory.get_full_steps()
    def get_react_time_efficiency(self):
        timing = []
        for step in self.agent.memory.get_full_steps():
            if step.get('duration'):
                timing.append(step.get('duration'))
        return pd.DataFrame.from_dict(
            data={"Steps":list(range(1,len(timing)+1)), "Time to Execute": timing}
        )
        
    