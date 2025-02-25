from typing import List, Union, Generator, Iterator
import os, json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


in_docker = True if os.getenv("IN_DOCKER", False) is not False else False
url = "http://host.docker.internal:11434" if in_docker else "http://localhost:11434"


class Pipeline:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="http://host.docker.internal:11434"
            if in_docker
            else "http://localhost:11434",
            description="Ollama API的基礎請求地址",
        )
        API_KEY: str = Field(default="ollama", description="用於身份驗證的API密鑰")
        MODEL: str = Field(
            default="deepseek-r1:1.5b",
            description="API请求的模型名称，默认为 deepseek-reasoner ",
        )
        MODEL_DISPLAY_NAME: str = Field(
            default="Ollama api model",
            description="模型名称，默认为 deepseek-reasoner-model",
        )
    
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "ollama_pipeline"
        self.name = "ollama pipeline"
        self.valves = self.Valves()
    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        model    = self.valves.MODEL
        base_url = self.valves.BASE_URL

        if "user" in body:
            print("\n\n\n\n######################################")
            print(f"# User: {body['user']['name']} ({body['user']['id']})")
            print(f"# Message: {user_message}")
            print(f"# Model: {model}")
            print(json.dumps(body))
            print("\n\n\n\n######################################")
        try:
            llm = ChatOllama(model=model, base_url=base_url)
            res = llm.stream(user_message) | StrOutputParser()
            model_integration = llm.__class__.__name__

            if model_integration.startswith("Chat"):
                return (r.content for r in res)
            else:
                return res

        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    Pipe = Pipeline()

    data = {
        "stream": False,
        "model": "ollama_openai_compatibility",
        "messages": [
            {
                "role": "user",
                "content": '### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history.\n### Guidelines:\n- The title should clearly represent the main theme or subject of the conversation.\n- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.\n- Write the title in the chat\'s primary language; default to English if multilingual.\n- Prioritize accuracy over excessive creativity; keep it clear and simple.\n### Output:\nJSON format: { "title": "your concise title here" }\n### Examples:\n- { "title": "\ud83d\udcc9 Stock Market Trends" },\n- { "title": "\ud83c\udf6a Perfect Chocolate Chip Recipe" },\n- { "title": "Evolution of Music Streaming" },\n- { "title": "Remote Work Productivity Tips" },\n- { "title": "Artificial Intelligence in Healthcare" },\n- { "title": "\ud83c\udfae Video Game Development Insights" }\n### Chat History:\n<chat_history>\nUSER: \u4ecb\u7d39\u7f8e\u570b\u6b77\u53f2\nASSISTANT: I\'m sorry, but the provided context doesn\'t include any information related to American history.\n</chat_history>',
            }
        ],
        "user": {
            "name": "arthur",
            "id": "5c7bf72a-fe8c-42f8-979e-02a0ae1917b3",
            "email": "aaa890177@gmail.com",
            "role": "admin",
        },
        "max_tokens": 1000,
    }
    res = Pipe.pipe(
        "介紹美國歷史",
        "ollama",
        [],
        data,
    )
    for r in res:
        print(r, end="", flush=True)
