from langmem import create_manage_memory_tool, create_search_memory_tool, create_memory_manager
from pydantic import BaseModel, Field

class MemorySchema(BaseModel):
    thought:str
    reasoning:str
    action:str
    extracted_information:str



class MemoryModule:
    def __init__(self, llm):
        self.store = create_memory_manager(
            llm.llm,
            schemas=[MemorySchema],
            instructions="Extract your thoughts on the user query, any previous interactions and the question in front of you. Execute an action and provide your reasoning for executing such an action. Ensure all actions are sound, are related to the users query or will assist in resolving the users query. Penalize any other actions heavily. Finally if your action provides some useful information related to the user query, store it! You are encouraged to learn as much as possible.",
            enable_inserts=True,
            enable_updates=True

        )
        self.memory_manager = create_manage_memory_tool(
            namespace=("memories")
        )
        self.search_tool = create_search_memory_tool(
            namespace=("memories")
        )

    def get_react_tools(self):
        return [self.memory_manager, self.search_tool]


