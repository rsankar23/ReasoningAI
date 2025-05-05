#####################################################
# 这是用以辅助包装llm的





import os
import subprocess
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class BabelCloudRGB:
    def __init__(self, llm_rgb_path=None, agent=None):
        """
        初始化 BabelCloud LLM-RGB 评估器
        
        参数:
            llm_rgb_path: LLM-RGB 仓库的路径
            agent: 要评估的自反思代理
        """
        # 设置 LLM-RGB 路径
        if llm_rgb_path is None:
            self.llm_rgb_path = Path(os.getcwd()) / "LLM-RGB"
        else:
            self.llm_rgb_path = Path(llm_rgb_path)
        
        # 保存代理引用
        self.agent = agent
        
        # 确保 LLM-RGB 仓库已克隆
        self._ensure_llm_rgb_repo()
        
        # 设置输出目录
        self.output_dir = Path(os.getcwd()) / "evaluation_results"
        self.output_dir.mkdir(exist_ok=True)
    
    # ... [其他方法保持不变] ...
    
    def create_agent_provider(self):
        """创建一个可用于 promptfoo 的自定义提供商配置"""
        # 创建一个临时脚本，将您的代理包装为 promptfoo 提供商
        provider_script = self.llm_rgb_path / "custom_provider.js"
        
        # 编写包装脚本
        with open(provider_script, 'w') as f:
            f.write("""
            const { spawn } = require('child_process');
            
            module.exports = {
              id: 'reflection-agent',
              validate() {
                return true;
              },
              async callApi(prompt, options) {
                return new Promise((resolve, reject) => {
                  // 调用 Python 脚本，该脚本会调用您的自反思代理
                  const process = spawn('python', ['agent_wrapper.py', prompt]);
                  
                  let output = '';
                  process.stdout.on('data', (data) => {
                    output += data.toString();
                  });
                  
                  process.stderr.on('data', (data) => {
                    console.error(`Error: ${data}`);
                  });
                  
                  process.on('close', (code) => {
                    if (code === 0) {
                      resolve({ output });
                    } else {
                      reject(new Error(`Agent process exited with code ${code}`));
                    }
                  });
                });
              }
            };
            """)
        
        # 创建一个 Python 包装脚本，调用您的代理
        wrapper_script = self.llm_rgb_path / "agent_wrapper.py"
        
        with open(wrapper_script, 'w') as f:
            f.write("""
            import sys
            import json
            import os
            
            # 添加父目录到路径，以便导入您的代理模块
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from reflection import build_self_reflective_agent
            from llm import LLM
            from langchain_tavily import TavilySearch
            from langchain_core.prompts import PromptTemplate
            
            # 初始化必要组件
            def get_agent():
                tool = TavilySearch(max_results=10)
                template = \"\"\"
                    Answer the following questions as best you can...
                    [您的原始模板内容]
                \"\"\"
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["input", "agent_scratchpad"],
                    partial_variables={"tool_names": tool.name, "tools": [tool]}
                )
                llm = LLM(source="Anthropic")  # 或其他您正在使用的模型
                return build_self_reflective_agent(llm, tool, prompt, max_reflection_turns=3)
            
            def main():
                if len(sys.argv) < 2:
                    print("Missing prompt argument")
                    sys.exit(1)
                
                prompt = sys.argv[1]
                
                # 获取代理
                agent = get_agent()
                
                # 执行代理
                response, reflection, trace = agent(prompt)
                
                # 输出最终结果
                print(response.get("output", ""))
            
            if __name__ == "__main__":
                main()
            """)
        
        return "reflection-agent"
    
    def update_config_for_agent(self):
        """更新配置文件以使用自定义代理"""
        config_path = self.llm_rgb_path / "promptfooconfig.yaml"
        
        # 读取现有配置
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 创建自定义提供商文件路径
        provider_js_path = "/content/ReasoningAI/LLM-RGB/custom_provider.js"
        
        # 正确的自定义提供商格式
        config['providers'] = [
            {
                "custom": {  # 这里需要使用 "custom" 键
                    "id": "reflection-agent",
                    "module": provider_js_path
                }
            }
        ]
        
        # 保存配置
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
    def run_agent_evaluation(self):
        """运行针对自反思代理的评估"""
        # 更新配置以使用自定义代理
        self.update_config_for_agent()
        
        # 执行评估
        st.info("Running LLM-RGB evaluation of self-reflective agents...")
        result = subprocess.run(
            ["npm", "run", "eval"],
            cwd=str(self.llm_rgb_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            st.error(f"Evaluation Failure: {result.stderr}")
            return None
        
        # 生成报告
        st.info("Generating report...")
        result = subprocess.run(
            ["npm", "run", "render"],
            cwd=str(self.llm_rgb_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            st.error(f"Report generation failed: {result.stderr}")
            return None
        

        experiment_dirs = list(self.output_dir.glob("*"))
        if not experiment_dirs:
            st.error("No evaluation results found")
            return None
            
        latest_experiment = max(experiment_dirs, key=os.path.getctime)
        
        return latest_experiment

    def _ensure_llm_rgb_repo(self):
    #"""确保 LLM-RGB 仓库存在，不存在则克隆"""
      if not self.llm_rgb_path.exists():
          st.info("Cloning LLM-RGB repository, please wait...")
          subprocess.run(
              ["git", "clone", "https://github.com/babelcloud/LLM-RGB.git", str(self.llm_rgb_path)],
              check=True
          )
          
          # 安装依赖
          st.info("Installing dependencies...")
          subprocess.run(
              ["npm", "install"], 
              cwd=str(self.llm_rgb_path),
              check=True
          )
          
          # 准备配置文件
          if not (self.llm_rgb_path / "promptfooconfig.yaml").exists():
              subprocess.run(
                  ["cp", "promptfooconfig.yaml.example", "promptfooconfig.yaml"],
                  cwd=str(self.llm_rgb_path),
                  check=True
              )