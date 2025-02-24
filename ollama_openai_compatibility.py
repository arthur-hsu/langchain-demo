from typing import List, Union, Generator, Iterator
from langchain_openai import ChatOpenAI, OpenAI
import os, json
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

in_docker = True if os.getenv("IN_DOCKER", False) is not False else False
url = "http://host.docker.internal:11434" if in_docker else "http://localhost:11434"






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
        index_params       = {"index_type": "FLAT", "metric_type": "L2"},
    )

    # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(self, docs):
        d = 1
        for doc in docs:
            print(f"{d}: {'-'*100}")
            print(doc.page_content)
            print("\n")
            d+=1
        return "\n\n".join(doc.page_content for doc in docs)


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
        qa_chain = (
            {
                "context": self.vectorstore.as_retriever() | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
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

