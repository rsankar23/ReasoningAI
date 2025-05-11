import json
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from reflection import build_self_reflective_agent
from llm import LLM
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate


def setup_reflect_agent():
    """设置自反思代理"""
    tool = TavilySearch(max_results=10)
    template = PromptTemplate(
        template="""
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
        """,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"tool_names": tool.name, "tools": [tool]}
    )
    llm = LLM()
    return build_self_reflective_agent(llm, tool, template, max_reflection_turns=3)


def load_test_cases(base_dir="datasets/information_integration"):
    """
    加载信息整合测试集
    
    Args:
        base_dir: 测试集目录路径
    
    Returns:
        测试用例列表
    """
    test_data = []
    test_file = os.path.join(base_dir, "test.json")
    
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_data = json.load(f)
    else:
        # 如果没有现有测试用例，则创建模拟数据
        test_data = create_mock_test_cases()
        
        # 确保目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        # 保存模拟数据以便将来使用
        with open(test_file, "w") as f:
            json.dump(test_data, f, indent=2)
    
    return test_data

def load_llm_rgb_testcases(testcase_dir="LLM-RGB/testcases", random_sample=None):
    """
    加载LLM-RGB的测试用例，并可选择随机抽样
    
    Args:
        testcase_dir: 测试用例目录路径
        random_sample: 随机抽样的数量，None表示加载全部
    
    Returns:
        测试用例列表
    """
    test_data = []
    
    # 检查目录是否存在
    if not os.path.exists(testcase_dir):
        print(f"Warning: Testcase directory {testcase_dir} not found.")
        return create_mock_test_cases()
    
    # 获取所有配置文件
    config_files = [f for f in os.listdir(testcase_dir) if f.endswith('_config.yaml')]
    
    if not config_files:
        print("No config files found in testcase directory.")
        return create_mock_test_cases()
    
    # 如果需要随机抽样，对配置文件列表进行随机抽样
    if random_sample is not None and random_sample < len(config_files):
        import random
        config_files = random.sample(config_files, random_sample)
        print(f"Randomly sampled {random_sample} test cases.")
    
    import yaml
    
    for config_file in config_files:
        test_id = config_file.replace('_config.yaml', '')
        config_path = os.path.join(testcase_dir, config_file)
        prompt_path = os.path.join(testcase_dir, f"{test_id}_prompt.txt")
        
        # 跳过没有对应提示文件的配置
        if not os.path.exists(prompt_path):
            continue
        
        # 读取配置文件
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                if not config or not isinstance(config, list) or len(config) == 0:
                    continue
                
                test_config = config[0]  # 通常配置是列表中的第一项
            except Exception as e:
                print(f"Error parsing config file {config_file}: {e}")
                continue
        
        # 读取提示文件
        with open(prompt_path, 'r') as f:
            prompt_text = f.read()
        
        # 提取测试用例信息
        test_case = {
            "id": test_id,
            "question": prompt_text,  # 使用整个提示作为问题
            "context": test_config.get('description', ''),
            "difficulties": test_config.get('vars', {}).get('difficulties', {}),
            "assertions": test_config.get('assert', [])
        }
        
        # 从断言中提取期望答案(如果存在)
        expected_answer = ""
        for assertion in test_case["assertions"]:
            if assertion.get('type') == 'equals' or assertion.get('type') == 'contains-any':
                values = assertion.get('value', [])
                if isinstance(values, list):
                    expected_answer = " ".join(values)
                else:
                    expected_answer = str(values)
                break
        
        test_case["answer"] = expected_answer
        test_data.append(test_case)
    
    print(f"Loaded {len(test_data)} test cases from LLM-RGB.")
    return test_data

def create_mock_test_cases():
    """
    创建模拟测试用例，结构类似于LLM-RGB格式
    """
    # 基于LLM-RGB测试用例格式创建模拟数据
    test_cases = [
        {
            "id": "info_integration_1",
            "question": "What is the capital of France and what is its population?",
            "context": "This question tests your ability to integrate information from different sources.",
            "answer": "paris",
            "difficulties": {
                "context-length": 1,
                "reasoning-depth": 2,  
                "instruction-compliance": 1
            }
        },
        {
            "id": "info_integration_2",
            "question": "Who was Albert Einstein and what is his most famous equation?",
            "context": "This question tests your ability to integrate biographical and scientific information.",
            "answer": "e=mc2",
            "difficulties": {
                "context-length": 1,
                "reasoning-depth": 2,
                "instruction-compliance": 1
            }
        },
        {
            "id": "info_integration_3",
            "question": "Compare and contrast renewable and non-renewable energy sources.",
            "context": "This question tests your ability to analyze different categories and make comparisons.",
            "answer": "renewable sustainable, non-renewable finite",
            "difficulties": {
                "context-length": 1,
                "reasoning-depth": 3,
                "instruction-compliance": 1
            }
        },
        {
            "id": "info_integration_4",
            "question": "Explain the process of photosynthesis and its importance for life on Earth.",
            "context": "This question tests your understanding of biological processes and ecological relationships.",
            "answer": "plants convert sunlight carbon dioxide water oxygen glucose",
            "difficulties": {
                "context-length": 1,
                "reasoning-depth": 3,
                "instruction-compliance": 1
            }
        },
        {
            "id": "info_integration_5",
            "question": "Describe the main causes and effects of climate change.",
            "context": "This question tests your ability to understand complex cause-and-effect relationships.",
            "answer": "greenhouse gases human activities global warming",
            "difficulties": {
                "context-length": 1,
                "reasoning-depth": 3,
                "instruction-compliance": 1
            }
        }
    ]
    return test_cases


def evaluate_agent_on_testcases(test_data, max_samples=None, save_path="eval_results.csv"):
    """
    使用LLM-RGB格式测试用例评估自反思代理
    
    Args:
        test_data: 测试用例列表
        max_samples: 最大样本数量，None表示全部
        save_path: 结果保存路径
        
    Returns:
        元组(准确率, 错误样本)
    """
    reflect_and_react = setup_reflect_agent()
    
    total = len(test_data) if max_samples is None else min(len(test_data), max_samples)
    correct = 0
    wrong_samples = []
    all_results = []
    
    print(f"Evaluating on {total} samples...")
    
    for i in tqdm(range(total)):
        item = test_data[i]
        
        # 提取测试用例信息
        question = item.get("question", "")
        context = item.get("context", "")
        expected_answer = item.get("answer", "").lower()
        difficulties = item.get("difficulties", {})
        test_id = item.get("id", f"test_{i}")
        
        # 构建输入文本
        input_text = f"{context}\n\nQuestion: {question}" if context else question
        
        try:
            # 调用代理并收集反思信息
            response, reflection, trace, reflections = reflect_and_react(input_text, collect_reflections=True)
            output = response.get("output", "").lower()
            
            # 检查答案是否正确(简单包含匹配)
            is_correct = expected_answer in output
            num_rounds = len(reflections)
            difficulty_score = sum(difficulties.values()) if difficulties else 0
            
            if is_correct:
                correct += 1
            else:
                wrong_samples.append({
                    "id": test_id,
                    "question": question,
                    "context": context,
                    "expected": expected_answer,
                    "output": output,
                    "reflection": reflection,
                    "reflection_rounds": num_rounds,
                    "difficulty_score": difficulty_score
                })
            
            # 收集评估数据
            all_results.append({
                "id": test_id,
                "question": question,
                "context": context[:50] + "..." if len(context) > 50 else context,
                "expected": expected_answer,
                "output": output[:100] + "..." if len(output) > 100 else output,
                "is_correct": is_correct,  # Changed from 'correct' to 'is_correct'
                "reflection_rounds": num_rounds,
                "difficulty_score": difficulty_score,
                "reasoning_depth": difficulties.get("reasoning-depth", 0) if difficulties else 0
            })
            
        except Exception as e:
            print(f"[{i+1}] Error in test case {test_id}: {str(e)}")
    
    # 保存结果到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"📁 Saved evaluation results to {save_path}")
    
    # 创建评估报告
    generate_evaluation_report(df, "reflection_eval_report.html")
    
    # 绘制饼图
    plt.figure(figsize=(6, 6))
    plt.pie([correct, total - correct], labels=["Correct", "Wrong"], autopct='%1.1f%%', 
            colors=["#4CAF50", "#F44336"], explode=(0.1, 0))
    plt.title("Agent Accuracy on Information Integration Tasks")
    plt.savefig("accuracy_pie_chart.png")
    print(f"📊 Saved accuracy pie chart as accuracy_pie_chart.png")
    
    # 绘制反思轮数分布
    plt.figure(figsize=(10, 6))
    round_counts = df['reflection_rounds'].value_counts().sort_index()
    plt.bar(round_counts.index, round_counts.values, color='skyblue')
    plt.xlabel('Number of Reflection Rounds')
    plt.ylabel('Count')
    plt.title('Distribution of Reflection Rounds')
    plt.xticks(round_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("reflection_rounds_distribution.png")
    print(f"📊 Saved reflection rounds distribution as reflection_rounds_distribution.png")
    
    # 绘制准确率与推理深度的关系
    if 'reasoning_depth' in df.columns:
        plt.figure(figsize=(10, 6))
        accuracy_by_depth = df.groupby('reasoning_depth')['is_correct'].mean()  # Changed from 'correct' to 'is_correct'
        counts_by_depth = df.groupby('reasoning_depth').size()
        
        plt.bar(accuracy_by_depth.index, accuracy_by_depth.values, color='lightgreen')
        plt.xlabel('Reasoning Depth')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Reasoning Depth')
        plt.ylim(0, 1)
        
        # 添加样本数量标签
        for depth, acc in zip(accuracy_by_depth.index, accuracy_by_depth.values):
            count = counts_by_depth[depth]
            plt.text(depth, acc + 0.02, f"n={count}", ha='center')
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("accuracy_by_reasoning_depth.png")
        print(f"📊 Saved accuracy by reasoning depth as accuracy_by_reasoning_depth.png")
    
    accuracy = correct / total if total > 0 else 0
    print(f"✅ Overall accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy, wrong_samples


def generate_evaluation_report(df, output_path="reflection_eval_report.html"):
    """
    生成HTML格式评估报告
    
    Args:
        df: 评估结果数据框
        output_path: 输出文件路径
    """
    # 计算总体统计
    total_cases = len(df)
    correct_cases = df['is_correct'].sum()  # Changed from 'correct' to 'is_correct'
    accuracy = correct_cases / total_cases if total_cases > 0 else 0
    avg_reflection_rounds = df['reflection_rounds'].mean()
    
    # 生成HTML报告
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Reflection Agent Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-item {{ margin: 10px 0; }}
        .correct {{ color: green; }}
        .incorrect {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ text-align: left; padding: 12px; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        .details-btn {{ 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            padding: 5px 10px; 
            cursor: pointer; 
            border-radius: 3px;
        }}
        .details-content {{ 
            display: none; 
            background-color: #f9f9f9; 
            padding: 10px; 
            border: 1px solid #ddd;
            margin-top: 5px;
            white-space: pre-wrap;
        }}
    </style>
    <script>
        function toggleDetails(id) {{
            var content = document.getElementById(id);
            if (content.style.display === "block") {{
                content.style.display = "none";
            }} else {{
                content.style.display = "block";
            }}
        }}
    </script>
</head>
<body>
    <h1>Reflection Agent Evaluation Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p class="summary-item"><strong>Total Test Cases:</strong> {total_cases}</p>
        <p class="summary-item"><strong>Correct Answers:</strong> {correct_cases}</p>
        <p class="summary-item"><strong>Accuracy:</strong> {accuracy:.2%}</p>
        <p class="summary-item"><strong>Average Reflection Rounds:</strong> {avg_reflection_rounds:.2f}</p>
    </div>
    
    <h2>Test Cases Results</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Question</th>
            <th>Result</th>
            <th>Reflection Rounds</th>
            <th>Difficulty Score</th>
            <th>Details</th>
        </tr>
    """
    
    for i, row in df.iterrows():
        result_class = "correct" if row['is_correct'] else "incorrect"  # Changed from 'correct' to 'is_correct'
        result_text = "✓ Correct" if row['is_correct'] else "✗ Incorrect"  # Changed from 'correct' to 'is_correct'
        
        html_content += f"""
        <tr>
            <td>{row['id']}</td>
            <td>{row['question']}</td>
            <td class="{result_class}">{result_text}</td>
            <td>{row['reflection_rounds']}</td>
            <td>{row.get('difficulty_score', 'N/A')}</td>
            <td>
                <button class="details-btn" onclick="toggleDetails('details-{i}')">Show/Hide</button>
                <div id="details-{i}" class="details-content">
                    <strong>Expected:</strong> {row['expected']}
                    <br><br>
                    <strong>Output:</strong> {row['output']}
                </div>
            </td>
        </tr>
        """
    
    html_content += """
    </table>
</body>
</html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Generated evaluation report at {output_path}")


if __name__ == "__main__":
    # 加载测试用例   
    test_data = load_llm_rgb_testcases()

    # 评估代理
    evaluate_agent_on_testcases(test_data, max_samples=None)