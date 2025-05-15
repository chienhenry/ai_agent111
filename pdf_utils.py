from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
import os
import tempfile
import uuid


def qa_agent(deepseek_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com/v1",
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

        # 创建问答链并获取回答
        qa = ConversationalRetrievalChain.from_llm(
            llm=model, retriever=retriever, memory=memory
        )
        response = qa.invoke({"chat_history": memory, "question": question})
        return response

    finally:
        # 确保临时文件被删除
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass
