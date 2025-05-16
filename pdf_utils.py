"""
PDF文件问答工具模块

本模块提供了基于PDF文件的智能问答功能，使用LangChain和Deepseek模型来解析PDF内容，并回答用户提问。
使用LangChain LCEL (LangChain Expression Language)实现，确保与最新版本兼容。

主要功能：
1. PDF文件加载和分块
2. 向量存储和检索
3. 基于对话历史的问答
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
import config
import os
import tempfile
import uuid


def qa_agent(deepseek_api_key, memory, uploaded_file, question):
    # 初始化LLM
    model = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com/v1",
        request_timeout=60.0,
    )

    # 使用临时文件夹和唯一文件名避免权限冲突
    temp_dir = tempfile.gettempdir()
    unique_filename = f"pdf_temp_{uuid.uuid4().hex}.pdf"
    temp_file_path = os.path.join(temp_dir, unique_filename)

    try:
        # 读取上传的文件内容
        file_content = uploaded_file.read()

        # 写入临时文件
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # 处理PDF文件
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", "。", "！", "？", "，", "、", ""],
        )
        texts = text_splitter.split_documents(docs)

        # 创建嵌入和向量数据库
        embeddings_model = BaichuanTextEmbeddings(
            model="Baichuan-Text-Embedding", api_key=config.BAICHUAN_API_KEY
        )
        db = FAISS.from_documents(texts, embeddings_model)
        retriever = db.as_retriever()

        # 定义提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业PDF文档问答助手，根据以下上下文和对话历史回答问题：\n上下文：{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        # 创建LCEL链
        chain = (
            {
                "context": lambda x: retriever.get_relevant_documents(x["question"]),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | model
            | StrOutputParser()
        )

        # 处理历史消息
        chat_history = []
        if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
            chat_history = memory.chat_memory.messages

        # 调用链
        response = chain.invoke({"question": question, "chat_history": chat_history})

        # 添加新消息到历史记录
        from langchain_core.messages import AIMessage, HumanMessage

        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response)

        # 返回与之前格式相同的结果
        return {
            "answer": response,
            "chat_history": memory.chat_memory.messages,
            "source_documents": retriever.get_relevant_documents(question),
        }

    finally:
        # 确保临时文件被删除
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass
