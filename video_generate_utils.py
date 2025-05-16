from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# import os


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


def generate_script(subject, video_length, creativity, api_key):
    title_template = ChatPromptTemplate.from_messages(
        [("human", "请为'{subject}'这个主题的视频想一个吸引人的标题")]
    )
    script_template = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
             视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
             要求开头抓住限球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
             整体内容的表达方式要尽量轻松有趣，吸引年轻人。
             脚本内容可以结合以下维基百科搜索出的信息，但仅作为参考，只结合相关的即可，对不相关的进行忽略：
             ```{wikipedia_search}```""",
            )
        ]
    )

    try:
        # 使用标准参数创建模型，避免可能导致验证错误的参数
        model = ChatOpenAI(
            model_name="deepseek-chat",  # 使用model_name代替model
            openai_api_key=api_key,  # 使用openai_api_key代替api_key
            temperature=float(creativity),  # 确保temperature是浮点数
            openai_api_base="https://api.deepseek.com/v1",
            request_timeout=60.0,  # 增加超时时间并确保是浮点数
        )

        title_chain = title_template | model
        script_chain = script_template | model

        # 使用重试机制获取标题
        title_response = invoke_with_retry(title_chain, {"subject": subject})
        title = (
            title_response.content
            if hasattr(title_response, "content")
            else str(title_response)
        )

        search = WikipediaAPIWrapper(lang="zh")
        search_result = search.run(subject)

        # 使用重试机制获取脚本
        script_response = invoke_with_retry(
            script_chain,
            {
                "title": title,
                "duration": video_length,
                "wikipedia_search": search_result,
            },
        )
        script = (
            script_response.content
            if hasattr(script_response, "content")
            else str(script_response)
        )

        return search_result, title, script

    except Exception as e:
        # 捕获任何异常并提供友好的错误消息
        error_message = str(e)
        print(f"生成脚本时出错: {error_message}")
        return (
            f"无法获取维基百科内容: {error_message}",
            f"无法生成标题: {error_message}",
            f"无法生成脚本。请稍后再试或尝试不同的主题。错误信息: {error_message}",
        )


# print(generate_script("sora模型", 1, 0.7, os.getenv("OPENAI_API_KEY")))
