from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 設定 PDF 來源
pdf_path = "./resource/ts001-1-0-4-lorawan-l2-1-0-4-specification.pdf"
pdf_loader = PyMuPDFLoader(pdf_path)
documents = pdf_loader.load()

# 設定文字切割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 設定 embedding model
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': True}

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 轉換文本為 embedding
texts = [doc.page_content for doc in docs]
embeddings = embedding_model.embed_documents(texts)

# 連接到 Milvus
connections.connect(host="localhost", port="19530")

# 定義 collection 名稱
collection_name = "pdf_embeddings"

# 如果 collection 存在，則刪除
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' 已刪除")

# 定義 Collection Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
]

schema = CollectionSchema(fields, description="PDF Embeddings Collection")

# 創建 Collection
collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' 已建立")

# 建立索引（使用 HNSW 來提高查詢效能）
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",  # 或者 "IVF_PQ" 也可以
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("embedding", index_params)
print("索引已建立")

# 插入數據（需要拆分為列表格式）
entities = [
    embeddings,
    texts
]


collection.insert(entities)
collection.flush()  # 確保數據寫入
collection.load()
print(f"Collection '{collection_name}' 已載入，資料已成功插入！")

