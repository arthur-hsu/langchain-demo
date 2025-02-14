




from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
tokens = {}
if os.path.exists('.env'):
    with open('.env', 'r') as token_file:
        tokens = {name: var for name, var in (line.replace('\n','').split('=', 1) for line in token_file)}
api_key = os.getenv("GEMINI_API_KEY", tokens.get('GEMINI_API_KEY'))
# model="gemini-2.0-flash"
# base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
model="deepseek-r1:32b"
base_url="http://host.docker.internal:11434"


class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "langchain_llm_pipeline"
        self.name = "langchain llm pipeline"

        # Initialize rate limits
        self.valves = self.Valves(**{"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")})

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")
        print(f'user_message:{user_message}')
        print(f'model_id:{model_id}')
        print(f'messages:{messages}')
        print(f'body:{body}')

        # llm = ChatOpenAI(model=model,api_key=api_key,base_url=base_url)
        # res = llm.invoke("Sing a ballad of LangChain.")
        # print(res.content)
        if body.get("title", False):
            print("Title Generation")
            return "langchain llm pipeline"
        else:
            # llm = ChatOpenAI(model=model,api_key='',base_url=base_url)
            llm = OllamaLLM(model=model, base_url=base_url)
            llm_response = llm.invoke(user_message)
            # print(llm_response.pretty_print())
            context = llm_response.content if hasattr(llm_response, "content") else llm_response
            if not isinstance(context, str):
                context = context.decode("utf-8")

            return context if context else "No information found"

if __name__ == "__main__":
    pipeline = Pipeline()
    query = "Who are you?"
    answer = pipeline.pipe(query,'1',[{}],{})
    print("Answer:", answer)
