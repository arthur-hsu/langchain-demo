from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAI
import os
from langchain.chains import RetrievalQA

# 1. 設定 Milvus 伺服器
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "pdf_embeddings"

# 2. 設定 Hugging Face Embeddings
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': True}

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 3. 連接 Milvus 向量資料庫
vector_db = Milvus(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    connection_args={"uri": MILVUS_URI},
)

query_text = "LoRaWan Class B 介紹"
docs = vector_db.similarity_search(query_text, k=5)  # 取前 5 筆最相關的結果

# 印出結果
print("\n🔍 查詢結果：")
for i, doc in enumerate(docs):
    print(f"📄 {i+1}: {doc.page_content}\n")



# in_docker = True if os.getenv("IN_DOCKER", False) is not False else False
# url = "http://host.docker.internal:11434" if in_docker else "http://localhost:11434"
#
#
# # 4. 使用 LangChain OpenAI 調用本地 Ollama
# llm = ChatOpenAI(model="deepseek-r1:14b", base_url=f"{url}/v1", api_key="ollama",streaming=True)
#
#
# # 5. 建立 RAG 查詢流程
# retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # 取回 5 個最相關的內容
# rag_chain = RetrievalQA(llm=llm, retriever=retriever)
#
# # 6. 測試 RAG 問答
# query = "LoRaWan class b 介紹"
# result = rag_chain.run(query)
#
# print("RAG 生成的答案：", result)
#
