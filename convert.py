from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
from uuid import uuid4

from langchain_core.documents import Document


# 連接到 Milvus
connections.connect(host="localhost", port="19530")

# 定義 collection 名稱
collection_name = "pdf_embeddings"

# 如果 collection 存在，則刪除
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' 已刪除")



# 設定 PDF 來源
pdf_path = "./resource/ts001-1-0-4-lorawan-l2-1-0-4-specification.pdf"
documents = PyMuPDFLoader(pdf_path).load()

# 設定文字切割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)



embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    # model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress = True,
)



# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".

vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name=collection_name,
    # Set index_params if needed
    # index_params={"index_type": "FLAT", "metric_type": "L2"},
    index_params={"index_type": "FLAT", "metric_type": "IP"},  # 使用 內積(IP) 來匹配
    auto_id=True
)



vector_store.add_documents(docs)






















