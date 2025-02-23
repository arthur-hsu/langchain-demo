from langchain_huggingface import HuggingFaceEmbeddings

# 加載 bge-m3 模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
