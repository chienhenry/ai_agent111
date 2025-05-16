import streamlit as st

from langchain_community.chat_message_histories import ChatMessageHistory
from pdf_utils import qa_agent
import config


st.title("📑 AI智能PDF问答工具")

with st.sidebar:
    deepseek_api_key = st.text_input(
        "请输入DeepSeek API密钥：", value=config.DEEPSEEK_API_KEY, type="password"
    )
    st.markdown("[获取DeepSeek API key](https://platform.deepseek.com/api_keys)")

# 初始化聊天记忆
if "memory" not in st.session_state:
    memory_obj = type("obj", (object,), {"chat_memory": ChatMessageHistory()})
    st.session_state["memory"] = memory_obj

# 初始化聊天历史显示记录
if "chat_history_display" not in st.session_state:
    st.session_state["chat_history_display"] = []

uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")
question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)

if uploaded_file and question and not deepseek_api_key:
    st.info("请输入你的DeepSeek API密钥")

if uploaded_file and question and deepseek_api_key:
    with st.spinner("AI正在思考中，请稍等..."):
        try:
            response = qa_agent(
                deepseek_api_key, st.session_state["memory"], uploaded_file, question
            )

            st.write("### 答案")
            st.write(response["answer"])

            # 更新显示用的历史记录
            st.session_state["chat_history_display"].append(
                {"role": "user", "content": question}
            )
            st.session_state["chat_history_display"].append(
                {"role": "assistant", "content": response["answer"]}
            )
        except Exception as e:
            st.error(f"处理问题时出错: {str(e)}")
            st.info("请尝试重新提问或上传不同的PDF文件。")

# 显示聊天历史
if st.session_state["chat_history_display"]:
    with st.expander("历史消息"):
        for msg in st.session_state["chat_history_display"]:
            if msg["role"] == "user":
                st.markdown(f"**👤 问题**: {msg['content']}")
            else:
                st.markdown(f"**🤖 回答**: {msg['content']}")
            st.divider()
