from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction


def build_self_reflective_agent(llm, tool, prompt):
    agent = AgentExecutor(
        agent=create_react_agent(llm.llm, tools=[tool], prompt=prompt),
        tools=[tool],
        return_intermediate_steps=True,
        verbose=True,
        handle_parsing_errors=True
    )

    reflection_prompt = PromptTemplate.from_template("""
    You just generated the following reasoning steps and answer:

    {agent_trace}

    Reflect on this reasoning trace.
    - Were there any logical flaws, contradictions, or unsupported assumptions?
    - Was the final answer correct and well supported?

    If there was an issue, describe the mistake and suggest how to revise the reasoning. 
    If everything was fine, say "Looks good".

    Reflection:
    """)

    def reflect_and_react(input_question):
        # Step 1: Initial agent execution
        response = agent.invoke({"input": input_question})
        trace = response.get("intermediate_steps", [])

        # Step 2: Combine steps into a readable trace string
        trace_str = ""
        for step in trace:
            if isinstance(step, AgentAction):
                trace_str += f"Thought: {step.log}\nAction: {step.tool}\nAction Input: {step.tool_input}\n"
            else:
                trace_str += f"Observation: {step}\n"

        # Step 3: Reflect
        reflection = llm.invoke(reflection_prompt.format(agent_trace=trace_str))
        reflection_text = reflection.content.strip()

        # Step 4: Conditional re-run
        if "issue" in reflection_text.lower() or "mistake" in reflection_text.lower():
            second_response = agent.invoke({"input": input_question})
            return second_response, reflection_text, trace_str
        else:
            return response, reflection_text, trace_str

    return reflect_and_react
__all__ = ["build_self_reflective_agent"]
