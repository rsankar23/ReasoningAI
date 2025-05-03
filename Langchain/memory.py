# from langmem import create_manage_memory_tool, create_search_memory_tool, create_memory_manager
# from pydantic import BaseModel, Field

# class MemorySchema(BaseModel):
#     """Store all new facts, preferences, and relationships as triples."""
#     subject: str
#     predicate: str
#     object: str
#     context: str | None = None



# class MemoryModule:
#     def __init__(self, llm):
#         self.store = create_memory_manager(
#             llm.llm,
#             schemas=[MemorySchema],
#             instructions="Extract user preferences and any other useful information",
#             enable_inserts=True,
#             enable_updates=True

#         )
#         self.memory_manager = create_manage_memory_tool(
#             namespace=("memories")
#         )
#         self.search_tool = create_search_memory_tool(
#             namespace=("memories")
#         )

#     def get_react_tools(self):
#         return [self.memory_manager, self.search_tool]


