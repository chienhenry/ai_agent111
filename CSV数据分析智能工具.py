import pandas as pd
import streamlit as st
from csv_utils import dataframe_agent
import config

st.set_page_config(page_title="DIY AI工具集", page_icon="🤖", layout="centered")


def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)


st.title("💡 CSV数据分析智能工具")

with st.sidebar:
    deepseek_api_key = st.text_input(
        "请输入DeepSeek API密钥：", value=config.DEEPSEEK_API_KEY, type="password"
    )
    st.markdown("[获取DeepSeek API key](https://platform.deepseek.com/api_keys)")

data = st.file_uploader("上传你的数据文件（CSV格式）：", type="csv")
if data:
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])

query = st.text_area(
    "请输入你关于以上表格的问题，或数据提取请求，或可视化要求（支持散点图、折线图、条形图）："
)
button = st.button("生成回答")

if button and not deepseek_api_key:
    st.info("请输入你的DeepSeek API密钥")
if button and "df" not in st.session_state:
    st.info("请先上传数据文件")
if button and deepseek_api_key and "df" in st.session_state:
    with st.spinner("AI正在思考中，请稍等..."):
        response_dict = dataframe_agent(deepseek_api_key, st.session_state["df"], query)
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        if "table" in response_dict:
            st.table(
                pd.DataFrame(
                    response_dict["table"]["data"],
                    columns=response_dict["table"]["columns"],
                )
            )
        if "bar" in response_dict:
            create_chart(response_dict["bar"], "bar")
        if "line" in response_dict:
            create_chart(response_dict["line"], "line")
        if "scatter" in response_dict:
            create_chart(response_dict["scatter"], "scatter")

# columns = st.columns(4)
# # pages = {("Home","🏠"):'智能PDF问答工具.py',("Page1","1️⃣"):'视频脚本一键生成器.py',("Page2","2️⃣"):'克隆ChatGPT.py',("Page3","🌎"):'爆款小红书文案生成器.py'}
# # for i in range(4):
# columns[0].page_link("CSV数据分析智能工具.py", label="Home", icon="🏠")
# columns[1].page_link("pages/智能PDF问答工具.py", label="Page 1", icon="1️⃣")
# columns[2].page_link("pages/视频脚本一键生成器.py", label="Page 2", icon="2️⃣", disabled=True)
# columns[3].page_link("http://www.google.com", label="Google", icon="🌎")
# # st.page_link("CSV数据分析智能工具.py", label="Home", icon="🏠")
# # st.page_link("pages/智能PDF问答工具.py", label="Page 1", icon="1️⃣")
# # st.page_link("pages/视频脚本一键生成器.py", label="Page 2", icon="2️⃣", disabled=True)
# # st.page_link("http://www.google.com", label="Google", icon="🌎")
# pages = {
#      "Your account" : [
#          st.Page("CSV数据分析智能工具.py", title="Create your account"),
#          st.Page("pages/智能PDF问答工具.py", title="Manage your account")
#      ],
#      "Resources" : [
#          st.Page("pages/视频脚本一键生成器.py", title="Learn about us"),
#          st.Page("pages/克隆ChatGPT.py", title="Try it out")
#      ]
# }

# pg = st.navigation(pages)
# pg.run()
