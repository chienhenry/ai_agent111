import pandas as pd
import streamlit as st
from csv_utils import dataframe_agent
import config
import json

st.set_page_config(page_title="DIY AI工具集", page_icon="🤖", layout="centered")


def create_chart(input_data, chart_type):
    try:
        # 确保数据格式正确
        if not input_data.get("columns") or not input_data.get("data"):
            st.error("图表数据格式错误: 缺少columns或data字段")
            return

        # 检查数据是否为列表格式
        if not isinstance(input_data["data"], list):
            st.error("图表数据格式错误: data必须是列表")
            return

        # 格式化为正确的DataFrame格式
        # 检查是否为嵌套列表
        data = input_data["data"]
        columns = input_data["columns"]

        # 如果data是一维数组，将其转换为二维格式
        if data and not isinstance(data[0], list):
            # 对于职业类别和收入这样的数据
            if len(columns) == 2:
                # 常见情况：第一列是类别，第二列是数值
                df_data = pd.DataFrame(
                    {columns[1]: data},
                    index=(
                        columns[0].split(",")
                        if isinstance(columns[0], str)
                        else [columns[0]]
                    ),
                )
            elif len(columns) == 1:
                # 单列情况，使用序号作为索引
                df_data = pd.DataFrame({columns[0]: data})
            else:
                # 多列情况，数据可能是扁平的
                # 检查数据长度是否与列数匹配
                if len(data) % (len(columns) - 1) == 0:
                    # 尝试重塑数据
                    rows = len(data) // (len(columns) - 1)
                    reshaped_data = []
                    for i in range(rows):
                        row_data = data[
                            i * (len(columns) - 1) : (i + 1) * (len(columns) - 1)
                        ]
                        reshaped_data.append(row_data)
                    df_data = pd.DataFrame(reshaped_data, columns=columns[1:])
                    if isinstance(columns[0], list) and len(columns[0]) == rows:
                        df_data.index = columns[0]
                    else:
                        # 使用默认索引
                        pass
                else:
                    # 如果无法重塑，创建具有相同值的列
                    df_data = pd.DataFrame()
                    for i, col in enumerate(columns[1:], 1):
                        df_data[col] = data
        else:
            # 对于多维数组，正常创建DataFrame
            df_data = pd.DataFrame(data, columns=columns)
            # 设置第一列为索引，除非只有一列
            if len(columns) > 1:
                df_data.set_index(columns[0], inplace=True)

        # 创建图表
        if chart_type == "bar":
            st.bar_chart(df_data)
        elif chart_type == "line":
            st.line_chart(df_data)
        elif chart_type == "scatter":
            st.scatter_chart(df_data)
    except Exception as e:
        st.error(f"创建图表时出错: {str(e)}")
        # 输出当前数据格式以便调试
        st.error(f"数据格式: {input_data}")


st.title("💡 CSV数据分析智能工具")

with st.sidebar:
    deepseek_api_key = st.text_input(
        "请输入DeepSeek API密钥：", value=config.DEEPSEEK_API_KEY, type="password"
    )
    st.markdown("[获取DeepSeek API key](https://platform.deepseek.com/api_keys)")

data = st.file_uploader("上传你的数据文件（CSV格式）：", type="csv")
if data:
    try:
        st.session_state["df"] = pd.read_csv(data)
        with st.expander("原始数据"):
            st.dataframe(st.session_state["df"])
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")

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
        try:
            response_dict = dataframe_agent(
                deepseek_api_key, st.session_state["df"], query
            )

            # 处理回答
            if "answer" in response_dict:
                st.success("✅ 分析完成")
                st.write(response_dict["answer"])

            # 处理表格
            if "table" in response_dict and isinstance(response_dict["table"], dict):
                try:
                    st.table(
                        pd.DataFrame(
                            response_dict["table"]["data"],
                            columns=response_dict["table"]["columns"],
                        )
                    )
                except Exception as table_error:
                    st.error(f"显示表格时出错: {str(table_error)}")

            # 处理图表
            if "bar" in response_dict:
                create_chart(response_dict["bar"], "bar")
            if "line" in response_dict:
                create_chart(response_dict["line"], "line")
            if "scatter" in response_dict:
                create_chart(response_dict["scatter"], "scatter")

            # 如果没有任何内容显示，给用户反馈
            if not any(
                key in response_dict
                for key in ["answer", "table", "bar", "line", "scatter"]
            ):
                st.warning("AI未能生成有效的分析结果。请尝试重新提问或简化您的问题。")

        except Exception as e:
            st.error(f"处理您的请求时出错: {str(e)}")
            st.info(
                "请尝试重新提问或简化您的问题。如果问题持续，请检查数据格式或联系技术支持。"
            )

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
