import streamlit as st
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from llm import LLM
from langchain_tavily import TavilySearch
from langchain.memory import CombinedMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.memory.kg import ConversationKGMemory
import torch, os
from reflection import build_self_reflective_agent
import evaluate_rgb
from dotenv import load_dotenv

load_dotenv()

torch.classes.__path__ = [] # add this line to manually set it to empty. 

INDEX = 1
@st.cache_resource
def preprocessing():
    tool = TavilySearch(max_results = 10)
    template = """
            Answer the following questions as best you can. Do not search unrelated topics beyond the scope of the user query. You must be able to reasonably explain why the topic you searched was related to the user question. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}

    """
    prompt = PromptTemplate(
        template = template,
        input_variables = ["input", "agent_scratchpad"],
        partial_variables = {"tool_names":tool.name, "tools":[tool]}
    )
    # prompt = hub.pull("hwchase17/react")
    sources = ["HuggingFace", "Anthropic", "OpenAI"]
    llm = LLM(source=sources[INDEX])
    memory = ConversationSummaryMemory(llm=llm.llm)
    agent = AgentExecutor(
        agent = create_react_agent(
            llm.llm,
            tools = [tool],
            prompt = prompt
        ),
        return_intermediate_steps = True,
        tools = [tool],
        verbose = True,
        handle_parsing_errors=True,
        memory=memory
    )
    EXISTING_MEMORIES = ""
    return llm, agent, tool, prompt, memory,EXISTING_MEMORIES

llm, agent, tool, prompt, memory, EXISTING_MEMORIES = preprocessing()
reflect_and_react = build_self_reflective_agent(llm, tool, prompt,memory, max_reflection_turns=3)

st.title("Reasoning AI")
base_messages = []
with st.sidebar:
    models = st.selectbox("Models Available", ["HuggingFace", "Anthropic", "OpenAI"], index=INDEX)
    st.markdown("## 📊 Evaluate Agent on LLM-RGB")
    if st.button("Run RGB Evaluation (10 samples)", key="rgb_eval_btn"):
        with st.spinner("Running evaluation..."):
            try:
                data = evaluate_rgb.load_rgb_dataset()
                acc, wrongs = evaluate_rgb.evaluate_agent_on_rgb(data, max_samples=10)
                st.success(f"✅ Accuracy: {acc:.2%}")

                st.image("accuracy_pie_chart.png", caption="Accuracy Breakdown")

                if wrongs:
                    st.markdown("### ❌ Wrong Predictions")
                    for i, item in enumerate(wrongs):
                        with st.expander(f"Example {i+1}"):
                            st.markdown(f"""
                            **Question:** {item['question']}  
                            **Expected Answer:** {item['expected']}  
                            **Agent Output:** {item['output']}  
                            **Reflection:** {item['reflection']}  
                            """)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Enter your question here..."):
    # Display user message in chat message container

    with st.container():
        st.subheader("Base Model")
        with st.chat_message("user"):
            st.write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        resp = llm.invoke(user_prompt)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
        with st.chat_message("Base Model"):
            st.write(resp.content)
        

    with st.container():
        st.subheader("Out of Box ReAct Agent")
        with st.chat_message("user"):
            st.write(user_prompt)
        base_messages.append(HumanMessage(content=user_prompt))
        resp = agent.invoke({"input": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": resp.get("output")})
        base_messages.append(AIMessage(content=resp.get('output')))
        with st.chat_message("ReAct Agent"):
            st.write(resp.get("output"))
            with st.expander("Reasoning Steps:"):
                st.write(resp.get("intermediate_steps"))
            with st.expander("Memories Generated:"):
                new_summ = memory.predict_new_summary(messages=base_messages, existing_summary=EXISTING_MEMORIES)
                EXISTING_MEMORIES += f"\n\n{new_summ}"
                st.write(EXISTING_MEMORIES)
        
    with st.container():
        st.subheader("🧠 Self-Reflective Agent")

        final_response, final_reflection, final_trace = reflect_and_react(
            user_prompt, 
        )

        with st.chat_message("🧠 Self-Reflective Agent"):
            st.markdown("**🟡 Final Answer:**")
            st.markdown(final_response.get("output"))

        with st.expander("🔍 Reasoning Trace (Final Round)"):
            st.markdown(f"```text\n{final_trace}\n```")

        with st.expander("🪞 Reflection Rounds"):
            for r in all_reflections:
                with st.expander(f"🔁 Round {r['attempt']}"):
                    st.markdown("**Trace:**")
                    st.markdown(f"```text\n{r['trace']}\n```")
                    st.markdown("**Reflection:**")
                    st.markdown(r["reflection"])

__all__ = ["reflect_and_react"]