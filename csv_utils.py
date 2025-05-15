import json
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。

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


请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

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


def dataframe_agent(deepseek_api_key, df, query):
    # 配置超时时间更长的模型客户端
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        temperature=0,
        openai_api_base="https://api.deepseek.com/v1",
        request_timeout=60,  # 设置较长的超时时间
    )

    # 创建Pandas数据框代理
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent_executor_kwargs={"handle_parsing_errors": True},
        verbose=True,
        allow_dangerous_code=True,
    )

    # 构建完整提示
    prompt = PROMPT_TEMPLATE + query

    try:
        # 使用带重试的函数调用代理
        response = invoke_agent_with_retry(agent, prompt)
        # 解析JSON响应
        response_dict = json.loads(response["output"])
        return response_dict
    except Exception as e:
        # 捕获并处理所有异常
        error_message = str(e)
        print(f"数据分析出错: {error_message}")
        return {
            "answer": f"处理您的请求时出错。请尝试重新提问或简化您的问题。错误信息: {error_message}"
        }
