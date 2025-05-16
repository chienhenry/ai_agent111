from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI

import os
from langchain.memory import ConversationBufferMemory
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


# 定义重试装饰器
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_with_retry(chain, inputs):
    """带有重试机制的调用"""
    try:
        return chain.invoke(inputs)
    except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        print(f"API连接错误，正在重试: {str(e)}")
        # 强制引发异常以触发重试
        raise


def get_chat_response(prompt, memory, deepseek_api_key):
    try:
        model = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com/v1",
            request_timeout=60.0,
        )
        chain = ConversationChain(llm=model, memory=memory)

        # 使用重试机制调用LLM
        response = invoke_with_retry(chain, {"input": prompt})
        return response["response"]
    except Exception as e:
        error_message = str(e)
        print(f"聊天响应出错: {error_message}")
        return (
            f"处理您的请求时出错。请稍后再试或尝试不同的问题。错误信息: {error_message}"
        )


# memory = ConversationBufferMemory(return_messages=True)
# print(get_chat_response("牛顿提出过哪些知名的定律？", memory, os.getenv("OPENAI_API_KEY")))
# print(get_chat_response("我上一个问题是什么？", memory, os.getenv("OPENAI_API_KEY")))
