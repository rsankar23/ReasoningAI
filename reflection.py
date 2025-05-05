from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction
from langchain.memory import ConversationBufferMemory

def build_self_reflective_agent(llm, tool, prompt, max_reflection_turns=3):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )

    agent = AgentExecutor(
        agent=create_react_agent(llm.llm, tools=[tool], prompt=prompt),
        tools=[tool],
        memory=memory,  # ✅ 加入记忆
        return_intermediate_steps=True,
        verbose=True,
        handle_parsing_errors=True
    )

    reflection_prompt = PromptTemplate(
        template="""
    You just generated the following reasoning steps and answer:

    {agent_trace}

    Previous conversation:
    {chat_history}

    Reflect on this reasoning trace.
    - Were there any logical flaws, contradictions, or unsupported assumptions?
    - Was the final answer correct and well supported?

    If there was an issue, describe the mistake and suggest how to revise the reasoning. 
    If everything was fine, say "Looks good".

    Reflection:
    """,
        input_variables=["agent_trace", "chat_history"]
    )

    def reflect_and_react(input_question, collect_reflections=False):
        # 添加收集所有反思的容器（如果需要）
        all_reflections = [] if collect_reflections else None
        
        # 初始运行
        response = agent.invoke({"input": input_question})
        trace = response.get("intermediate_steps", [])

        # 格式化轨迹
        trace_str = ""
        for step in trace:
            if isinstance(step, AgentAction):
                trace_str += f"Thought: {step.log}\nAction: {step.tool}\nAction Input: {step.tool_input}\n"
            else:
                trace_str += f"Observation: {step}\n"

        # 反思
        reflection = llm.invoke(reflection_prompt.format(
            agent_trace=trace_str,
            chat_history=memory.load_memory_variables({})["chat_history"]
        ))        reflection_text = reflection.content.strip()
        
        # 记录这轮反思（如果需要）
        if collect_reflections:
            all_reflections.append({
                "attempt": 1,
                "trace": trace_str,
                "reflection": reflection_text
            })

        attempt = 1
        max_attempts = max_reflection_turns
        
        # 如果检测到问题，再次尝试
        while ("issue" in reflection_text.lower() or "mistake" in reflection_text.lower()) and attempt < max_attempts:
            attempt += 1
            response = agent.invoke({"input": input_question})
            new_trace = response.get("intermediate_steps", [])
            
            # 格式化新轨迹
            new_trace_str = ""
            for step in new_trace:
                if isinstance(step, AgentAction):
                    new_trace_str += f"Thought: {step.log}\nAction: {step.tool}\nAction Input: {step.tool_input}\n"
                else:
                    new_trace_str += f"Observation: {step}\n"
            
            # 反思新轨迹
            reflection = llm.invoke(reflection_prompt.format(agent_trace=new_trace_str))
            reflection_text = reflection.content.strip()
            
            # 更新轨迹
            trace_str = new_trace_str
            
            # 记录这轮反思（如果需要）
            if collect_reflections:
                all_reflections.append({
                    "attempt": attempt,
                    "trace": new_trace_str,
                    "reflection": reflection_text
                })
        
        if collect_reflections:
            return response, reflection_text, trace_str, all_reflections
        else:
            return response, reflection_text, trace_str

    return reflect_and_react
