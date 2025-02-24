from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from langchain_openai import ChatOpenAI, OpenAI
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
)

collection_name = "pdf_embeddings"
vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name=collection_name,
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)


in_docker = True if os.getenv("IN_DOCKER", False) is not False else False
url = "http://host.docker.internal:11434" if in_docker else "http://localhost:11434"


# 4. 使用 LangChain OpenAI 調用本地 Ollama
llm = ChatOpenAI(model="deepseek-r1:14b", base_url=f"{url}/v1", api_key="ollama",streaming=True)






# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    d = 1
    for doc in docs:
        print(f"{d}: {'-'*100}")
        print(doc.page_content)
        print("\n")
        d+=1
    return "\n\n".join(doc.page_content for doc in docs)


qa_chain = (
    {
        "context": vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

res = qa_chain.stream("LoRaWan class B introduce")
for r in res:
    print(r, end="", flush=True)
