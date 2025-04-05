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
    # def invoke(self, query):
    #     return self.executor.invoke({"input": query})



    #     self.prompt = PromptTemplate(
    #     template = "You are a digital assistant tasked with responding to users questions while referencing the ultimate resource; Wikipedia. Make the relevant focused topical searches based on the users query. Leverage the {tools} tools with the names {tool_names} at every interaction. The query is: {query}; Leverage the space here to provide your structured, reasoned thoughts, {agent_scratchpad}",
    #     input_variables = ["query", "agent_scratchpad"],
    #     partial_variables = {"tool_names":tool.name, "tools":[tool]}
    # )