import streamlit as st
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from llm import LLM
from langchain_tavily import TavilySearch
# from react_agent import ReActAgent
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
    working = "You are a digital assistant tasked with responding to users questions while referencing the ultimate resource; the internet. Make the relevant focused topical searches based on the users query. Leverage the {tools} tools with the names {tool_names} at every interaction. The query is: {query}; Leverage the space here to provide your structured, reasoned thoughts, {agent_scratchpad}"    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["input", "agent_scratchpad"],
        partial_variables = {"tool_names":tool.name, "tools":[tool]}
    )
    # prompt = hub.pull("hwchase17/react")
    sources = ["HuggingFace", "Anthropic", "OpenAI"]
    llm = LLM(source=sources[INDEX])
    agent = AgentExecutor(
        agent = create_react_agent(
            llm.llm,
            tools = [tool],
            prompt = prompt
        ),
        return_intermediate_steps = True,
            tools = [tool],
            verbose = True,
            handle_parsing_errors=True
    )
    return llm, agent, tool, prompt

llm, agent, tool, prompt = preprocessing()
reflect_and_react = build_self_reflective_agent(llm, tool, prompt, max_reflection_turns=3)

st.title("Reasoning AI")

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
        with st.chat_message("Base Model"):
            st.write(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})

    with st.container():
        st.subheader("Out of Box ReAct Agent")
        with st.chat_message("user"):
            st.write(user_prompt)
        resp = agent.invoke({"input": user_prompt})
        with st.chat_message("ReAct Agent"):
            st.write(resp.get("output"))
            with st.expander("Reasoning Steps:"):
                st.write(resp.get("intermediate_steps"))
        st.session_state.messages.append({"role": "assistant", "content": resp.get("output")})
    with st.container():
        st.subheader("🧠 Self-Reflective Agent")

        final_response, final_reflection, final_trace, all_reflections = reflect_and_react(
            user_prompt, collect_reflections=True
        )

        # 在这行代码后面立即添加调试输出
        # st.write("DEBUG - final_trace 类型:", type(final_trace))
        # st.write("DEBUG - final_trace 长度:", len(final_trace) if final_trace else 0)
        # st.write("DEBUG - all_reflections 类型:", type(all_reflections))
        # st.write("DEBUG - all_reflections 长度:", len(all_reflections) if all_reflections else 0)
        # if all_reflections and len(all_reflections) > 0:
        #     st.write("DEBUG - 第一个反思的trace长度:", len(all_reflections[0]['trace']) if all_reflections[0]['trace'] else 0)

        with st.chat_message("🧠 Self-Reflective Agent"):
            st.markdown("**🟡 Final Answer:**")
            st.markdown(final_response.get("output"))

        with st.expander("🔍 Reasoning Trace (Final Round)"):
            st.markdown(f"```text\n{final_trace}\n```")
        # 原来的样子
        # with st.expander("🪞 Reflection Rounds"):
        #     for r in all_reflections:
        #         with st.expander(f"🔁 Round {r['attempt']}"):
        #             st.markdown("**Trace:**")
        #             st.markdown(f"```text\n{r['trace']}\n```")
        #             st.markdown("**Reflection:**")
        #             st.markdown(r["reflection"])

        with st.expander("🪞 Reflection Rounds"):
            for r in all_reflections:
                st.markdown(f"### 🔁 Round {r['attempt']}")
                st.markdown("**Trace:**")
                st.markdown(f"```text\n{r['trace']}\n```")
                st.markdown("**Reflection:**")
                st.markdown(r["reflection"])
                st.markdown("---")  # 添加分隔线分隔不同轮次



__all__ = ["reflect_and_react"]
