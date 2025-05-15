import streamlit as st

from xiaohongshu_utils import generate_xiaohongshu
import config


st.header("爆款小红书AI写作助手 ✏️")
with st.sidebar:
    deepseek_api_key = st.text_input(
        "请输入DeepSeek API密钥：", value=config.DEEPSEEK_API_KEY, type="password"
    )
    st.markdown("[获取DeepSeek API密钥](https://platform.deepseek.com/api_keys)")

theme = st.text_input("主题")
submit = st.button("开始写作")

if submit and not deepseek_api_key:
    st.info("请输入你的DeepSeek API密钥")
    st.stop()
if submit and not theme:
    st.info("请输入生成内容的主题")
    st.stop()
if submit:
    with st.spinner("AI正在努力创作中，请稍等..."):
        result = generate_xiaohongshu(theme, deepseek_api_key)
    st.divider()
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("##### 小红书标题1")
        st.write(result.titles[0])
        st.markdown("##### 小红书标题2")
        st.write(result.titles[1])
        st.markdown("##### 小红书标题3")
        st.write(result.titles[2])
        st.markdown("##### 小红书标题4")
        st.write(result.titles[3])
        st.markdown("##### 小红书标题5")
        st.write(result.titles[4])
    with right_column:
        st.markdown("##### 小红书正文")
        st.write(result.content)
