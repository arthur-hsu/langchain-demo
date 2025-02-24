from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from langchain_openai import ChatOpenAI, OpenAI
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate



embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
)

collection_name = "pdf_embeddings"
vectorstore = Milvus(
    embedding_function=embedding_model,
    collection_name=collection_name,
    # index_params={"index_type": "FLAT", "metric_type": "L2"},
    index_params={"index_type": "FLAT", "metric_type": "IP"},  # 使用 內積(IP) 來匹配
)


in_docker = True if os.getenv("IN_DOCKER", False) is not False else False
url = "http://host.docker.internal:11434" if in_docker else "http://localhost:11434"


# 4. 使用 LangChain OpenAI 調用本地 Ollama
llm = ChatOpenAI(model="deepseek-r1:14b", base_url=f"{url}/v1", api_key="ollama",streaming=True)




class RetrieverPipeline:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore

        # **1. 設定 LLM 提示詞**
        self.question_prompt = PromptTemplate.from_template(
            "Analyze the following question, extract key terms, and rephrase it to optimize for vector database retrieval.\n"
            "Avoid overly verbose descriptions and ensure the query accurately matches relevant information.\n"
            "Original Question: {question}\n"
            "Optimized Query for Retrieval:"
        )

        # **2. 定義處理步驟**
        self.question_refiner = self.question_prompt | self.llm | StrOutputParser() | self.debug_and_return
        self.context_chain = self.question_refiner | RunnableLambda(self.debug_query) | self.format_docs

    def debug_and_return(self, question):
        """Debug: 輸出 LLM 優化後的問題"""
        print("\n🔍 LLM 思考後的問題:", question)
        return question

    def debug_query(self, q):
        """Debug: 輸出 LLM 優化後的查詢，並執行檢索"""
        print("\n🔍 Refined Query for Retrieval:", q)
        return self.vectorstore.as_retriever().invoke(q)

    def format_docs(self, docs):
        """格式化檢索回來的文件"""
        documents = ""
        for doc in docs:
            content = doc.page_content
            sources = f"來源: {doc.metadata.get('source', 'Unknown')} 第 {doc.metadata.get('page', 'N/A')} 頁"
            documents += f"{content}\n\n{sources}\n\n"
        return documents

    def retrieve_context(self, question):
        """執行 LLM 轉換問題 -> 檢索 -> 格式化文檔"""
        return self.context_chain.invoke(question)


# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
# prompt = hub.pull("rlm/rag-prompt")
# prompt = """
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Cite the sources in parentheses after each relevant statement.
#
# Question: {question}  
# Context: {context}  
# Answer:
# """
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on provided context. Your provided context can include text or tables, "
    "and may also contain semantic XML markup. Pay attention the semantic XML markup to understand more about the context semantics as "
    "well as structure (e.g. lists and tabular layouts expressed with HTML-like tags)"
)
human_prompt = HumanMessagePromptTemplate.from_template(
    """Context:
    {context}
    Question: {question}"""
)
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])



retriever_pipeline = RetrieverPipeline(llm, vectorstore)
def print_prompt(q):
    print(f"Question: {q}\n")
    return q

# QA Chain
qa_chain = (
    {
        "question": RunnablePassthrough(),  # 保留原始問題
        "context": RunnableLambda(lambda q: retriever_pipeline.retrieve_context(q))
    }
    | prompt
    | print_prompt
    | llm
    | StrOutputParser()
)

# 執行查詢並輸出
res = qa_chain.stream("LoRaWan class B introduce")
for r in res:
    print(r, end="", flush=True)

