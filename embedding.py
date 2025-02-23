from langchain_huggingface import HuggingFaceEmbeddings

model_name="BAAI/bge-m3"
model_kwargs = {"device": "mps"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
embedding = hf.embed_query("hi this is harrison")
print(len(embedding))
print(embedding)
