from typing import List, Union, Generator, Iterator
from langchain_openai import ChatOpenAI, OpenAI
import os, json
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

in_docker = True if os.getenv("IN_DOCKER", False) is not False else False
url = "http://host.docker.internal:11434" if in_docker else "http://localhost:11434"





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



class Pipeline:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    collection_name = "pdf_embeddings"
    vectorstore = Milvus(
        embedding_function = embedding_model,
        collection_name    = collection_name,
        # index_params       = {"index_type": "FLAT", "metric_type": "L2"},
        index_params={"index_type": "FLAT", "metric_type": "IP"},  # 使用 內積(IP) 來匹配
    )

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

    class Valves(BaseModel):
        BASE_URL: str = Field(
            default = f"{url}/v1",
            description = "Ollama API的基礎請求地址"
        )
        API_KEY: str = Field(
            default = "ollama",
            description = "用於身份驗證的API密鑰"
        )
        MODEL: str = Field(
            default="deepseek-r1:14b",
            description="API请求的模型名称，默认为 deepseek-reasoner ",
        )
        MODEL_DISPLAY_NAME: str = Field(
            default="OpenAI api compatibility model",
            description="模型名称，默认为 deepseek-reasoner-model",
        )
    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass


    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "ollama_pipeline"
        self.name = "Ollama OpenAI compatibility"
        self.valves = self.Valves()

        

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        model    = self.valves.MODEL
        base_url = self.valves.BASE_URL
        api_key  = self.valves.API_KEY


        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print(f"# Model: {model}")
            print(json.dumps(body))
            print("######################################")
        
        llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,streaming=True)
        # result = llm.stream(user_message)
        def print_prompt(q):
            print(f"Question: {q}\n")
            return q

        # QA Chain
        qa_chain = (
            {
                "question": RunnablePassthrough(),  # 保留原始問題
                "context": RunnableLambda(lambda q: RetrieverPipeline(llm, self.vectorstore).retrieve_context(q))
            }
            | self.prompt
            | print_prompt
            | llm
            | StrOutputParser()
        )

        res = qa_chain.stream(user_message)
        return res




if __name__ == "__main__":
    Pipe = Pipeline()
    res = Pipe.pipe("你好", "ollama", [], {})
    for r in res:
        print(r, end="", flush=True)

