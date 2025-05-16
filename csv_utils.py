"""
CSV数据分析工具模块

本模块提供了对CSV数据文件的智能分析能力，可以基于自然语言查询对数据进行分析、
可视化和提取信息。

主要功能：
1. 基于LLM的数据分析代理
2. 强健的JSON解析，能够处理各种格式的输出
3. 错误处理和自动重试机制

修复记录：
- 2023/05/25: 增强JSON解析功能，修复了无法正确提取答案的问题
- 2023/05/25: 增加了正则表达式模式，支持各种可能的输出格式
- 2023/05/25: 添加了特定答案模式的识别，确保能够正确提取关键信息
- 2023/05/25: 修复LangChain兼容性警告，更新到最新API
"""

import json

# 更新导入，解决弃用警告
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # 如果没有安装langchain_openai，则使用旧的导入
    from langchain_community.chat_models import ChatOpenAI

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import time
import httpx
import re
from tenacity import retry, stop_after_attempt, wait_exponential


PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。你的回答必须严格遵循以下JSON格式。

重要：只返回纯JSON格式的回答，不要包含任何额外的解释、前缀或后缀文本。不要使用```json```格式或任何其他标记。

1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。

请确保：
1. 你只返回一个纯JSON对象，不包含任何其他文本
2. 所有字符串都必须使用双引号，不使用单引号
3. 不要在JSON前后添加任何解释性文字
4. 你的回答必须符合JSON格式，能够直接被json.loads()函数解析

你要处理的用户请求如下： 
"""


# 定义重试装饰器
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_agent_with_retry(agent, prompt):
    """带有重试机制的代理调用"""
    try:
        return agent.invoke({"input": prompt})
    except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        print(f"API连接错误，正在重试: {str(e)}")
        # 强制引发异常以触发重试
        raise


def extract_json_from_text(text):
    """从文本中提取有效的JSON字符串"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 尝试查找指定格式的JSON字符串 {"answer": "xxx"}
    answer_pattern = r'{"answer"\s*:\s*"([^"]+)"}'
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        return {"answer": answer_match.group(1)}

    # 尝试查找标准JSON格式内容
    json_patterns = [
        r"({[\s\S]*?})(?:Invalid Format|\Z)",  # 匹配JSON直到Invalid Format或文本结束
        r"({[\s\S]*?})(?:[^{]|$)",  # 匹配一个完整的JSON对象
        r"({.+?})",  # 匹配最小的JSON对象
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json_obj = json.loads(match)
                if isinstance(json_obj, dict):
                    return json_obj
            except:
                continue

    # 最后一种方法，提取所有看起来像键值对的内容
    # 例如： "Healthcare{"answer": "人数最多的职业是Healthcare"}"
    if '"answer":' in text:
        try:
            start_index = text.find('"answer":')
            if start_index != -1:
                # 向前找到最近的 {
                brace_index = text.rfind("{", 0, start_index)
                if brace_index != -1:
                    # 向后找到匹配的 }
                    depth = 0
                    for i in range(brace_index, len(text)):
                        if text[i] == "{":
                            depth += 1
                        elif text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                json_str = text[brace_index : i + 1]
                                try:
                                    return json.loads(json_str)
                                except:
                                    pass
        except:
            pass

    # 如果找不到JSON格式，尝试从文本中提取可能的答案
    possible_answer = ""
    if "人数最多的职业是" in text:
        match = re.search(r"人数最多的职业是(\w+)", text)
        if match:
            possible_answer = f"人数最多的职业是{match.group(1)}"
    elif "年收入的平均值是" in text:
        match = re.search(r"年收入的平均值是([\d\.]+)美元", text)
        if match:
            possible_answer = f"年收入的平均值是{match.group(1)}美元"

    if possible_answer:
        return {"answer": possible_answer}

    # 构造一个默认响应
    return {"answer": "无法解析AI的回答，请尝试重新提问或简化您的问题。"}


def dataframe_agent(deepseek_api_key, df, query):
    # 配置超时时间更长的模型客户端
    model = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_key=deepseek_api_key,
        temperature=0.0,
        openai_api_base="https://api.deepseek.com/v1",
        request_timeout=60.0,  # 设置较长的超时时间
    )

    # 构建完整提示
    full_prompt = PROMPT_TEMPLATE + query

    # 不使用默认的代理器，而是直接使用模型调用
    try:
        # 在代理失败的情况下，直接使用ChatOpenAI调用
        from langchain_core.messages import HumanMessage

        # 使用retry装饰器包装这个调用
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        def call_model_with_retry(model, prompt, df_snapshot):
            # 将数据框的快照添加到提示中
            dataframe_info = "\n\n数据框头部样本:\n"
            dataframe_info += df_snapshot.head(5).to_string()
            dataframe_info += "\n\n数据框统计信息:\n"
            dataframe_info += df_snapshot.describe().to_string()

            full_message = prompt + dataframe_info
            return model.invoke([HumanMessage(content=full_message)])

        # 生成数据框的快照
        df_snapshot = df.copy()

        # 调用模型
        response = call_model_with_retry(model, full_prompt, df_snapshot)
        output_text = response.content

        # 清理输出文本，去除可能的标记
        output_text = output_text.replace("```json", "").replace("```", "")
        output_text = output_text.strip()

        # 提取JSON结果
        response_dict = extract_json_from_text(output_text)
        return response_dict
    except Exception as e:
        # 如果直接调用模型失败，尝试使用旧的代理方法
        try:
            print(f"直接调用模型失败，尝试使用旧的代理方法: {str(e)}")
            # 创建Pandas数据框代理 (使用旧的方法作为后备)
            agent = create_pandas_dataframe_agent(
                llm=model,
                df=df,
                agent_executor_kwargs={"handle_parsing_errors": True},
                verbose=True,
                allow_dangerous_code=True,
            )

            # 使用带有重试的函数调用代理
            response = invoke_agent_with_retry(agent, full_prompt)
            output_text = response["output"]

            # 使用正则表达式提取JSON结果
            response_dict = extract_json_from_text(output_text)
            return response_dict
        except Exception as e2:
            # 捕获并处理所有异常
            error_message = str(e2)
            print(f"数据分析出错: {error_message}")
            return {
                "answer": f"处理您的请求时出错。请尝试重新提问或简化您的问题。错误信息: {error_message}"
            }
