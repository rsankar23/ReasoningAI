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

# 在 app.py 文件中添加以下导入
from babelcloud_rgb import BabelCloudRGB
import traceback

load_dotenv()


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
    
    # 添加测试用例数量选择
    num_test_cases = st.slider("Number of test cases", min_value=1, max_value=20, value=5, 
                            help="Select how many random test cases to evaluate")
    
    if st.button("Run RGB Evaluation", key="rgb_eval_btn"):
        with st.spinner(f"Running evaluation on {num_test_cases} random test cases..."):
            try:
                # 使用用户选择的数量随机抽样
                data = evaluate_rgb.load_llm_rgb_testcases(random_sample=num_test_cases)
                # 评估所有加载的测试用例
                acc, wrongs = evaluate_rgb.evaluate_agent_on_testcases(data, max_samples=None)
                
                st.success(f"✅ Accuracy: {acc:.2%}")

                # 显示生成的图表
                st.image("accuracy_pie_chart.png", caption="Accuracy Breakdown")
                
                # 如果生成了反思轮数分布图，显示它
                if os.path.exists("reflection_rounds_distribution.png"):
                    st.image("reflection_rounds_distribution.png", caption="Reflection Rounds Distribution")
                
                # 如果生成了推理深度准确率图，显示它
                if os.path.exists("accuracy_by_reasoning_depth.png"):
                    st.image("accuracy_by_reasoning_depth.png", caption="Accuracy by Reasoning Depth")
                
                # 提供HTML报告的链接
                if os.path.exists("reflection_eval_report.html"):
                    st.markdown("[📄 View Detailed Evaluation Report](reflection_eval_report.html)")

                if wrongs:
                    st.markdown("### ❌ Wrong Predictions")
                    for i, item in enumerate(wrongs):
                        with st.expander(f"Example {i+1}"):
                            st.markdown(f"""
                            **ID:** {item.get('id', 'Unknown')}  
                            **Question:** {item['question'][:100]}...  
                            **Expected Answer:** {item['expected']}  
                            **Agent Output:** {item['output'][:150]}...  
                            **Reflection Rounds:** {item['reflection_rounds']}  
                            **Difficulty Score:** {item.get('difficulty_score', 'N/A')}  
                            """)
                            with st.expander("Full Reflection"):
                                st.text(item['reflection'])
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

    st.markdown("---")
    st.markdown("## 🧠 评估自反思代理")
    st.markdown("使用 LLM-RGB 基准评估您的自反思代理实现")
    
    # 选择测试用例数量
    test_count = st.slider("测试用例数量", min_value=5, max_value=15, value=10)
    
    # 评估按钮
    if st.button("评估自反思代理", key="eval_reflection_agent"):
        with st.spinner("正在评估自反思代理..."):
            try:
                # 初始化 LLM-RGB 评估器，传入自反思代理
                evaluator = BabelCloudRGB(agent=reflect_and_react)
                
                # 运行针对代理的评估
                experiment_dir = evaluator.run_agent_evaluation()
                
                if experiment_dir:
                    # 解析结果
                    results = evaluator.parse_results(experiment_dir)
                    
                    # 显示结果
                    st.success("评估完成！")
                    
                    # 创建性能概览
                    st.subheader("自反思代理性能概览")
                    
                    # 这里需要根据实际结果格式调整
                    metrics = {
                        "推理深度": results.get("reasoning_depth", 0),
                        "自我反思次数": results.get("reflection_rounds", 0),
                        "正确性": results.get("accuracy", 0),
                        "总分": results.get("total_score", 0)
                    }
                    
                    # 显示指标
                    col1, col2 = st.columns(2)
                    for i, (metric, value) in enumerate(metrics.items()):
                        if i % 2 == 0:
                            col1.metric(metric, f"{value:.2f}")
                        else:
                            col2.metric(metric, f"{value:.2f}")
                    
                    # 显示详细评估结果
                    st.subheader("测试用例详细结果")
                    for i, test_case in enumerate(results.get("test_cases", [])):
                        with st.expander(f"测试用例 {i+1}: {test_case.get('name', '未命名')}"):
                            st.markdown(f"**提示**: {test_case.get('prompt', '')}")
                            st.markdown(f"**预期答案**: {test_case.get('expected', '')}")
                            st.markdown(f"**代理输出**: {test_case.get('output', '')}")
                            st.markdown(f"**反思过程**: {test_case.get('reflection', '')}")
                            st.markdown(f"**得分**: {test_case.get('score', 0)}/100")
                else:
                    st.error("评估未返回有效结果")
            except Exception as e:
                st.error(f"评估失败: {str(e)}")
                st.error(f"详细错误: {traceback.format_exc()}")

                
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


        with st.expander("🪞 Reflection Rounds"):
            for r in all_reflections:
                st.markdown(f"### 🔁 Round {r['attempt']}")
                st.markdown("**Trace:**")
                st.markdown(f"```text\n{r['trace']}\n```")
                st.markdown("**Reflection:**")
                st.markdown(r["reflection"])
                st.markdown("---")  # 添加分隔线分隔不同轮次

        with st.expander("🧠 Conversation Memory"):
            history_text = agent.memory.buffer_as_str()
            st.markdown(f"```text\n{history_text}\n```")





__all__ = ["reflect_and_react"]
