from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def qa_agent(deepseek_api_key, memory, uploaded_file, question):
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
    )
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""],
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = BaichuanTextEmbeddings(
        model="Baichuan-Text-Embedding", api_key=config.BAICHUAN_API_KEY
    )
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model, retriever=retriever, memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
