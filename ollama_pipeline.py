from typing import List, Union, Generator, Iterator
import os, json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaLLM
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
# # from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate


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

        model = self.valves.MODEL
        base_url = self.valves.BASE_URL

        if "user" in body:
            print("\n\n\n\n######################################")
            print(f"# User: {body['user']['name']} ({body['user']['id']})")
            print(f"# Message: {user_message}")
            print(f"# Model: {model}")
            print(json.dumps(body))
            print("\n\n\n\n######################################")
        try:
            """ This newest version of the pipeline is not working.
            llm = ChatOllama(model=model, base_url=base_url)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    ("human", "{input}"),
                    # Placeholders fill up a **list** of messages
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            tools = load_tools(["serpapi"])
            agent = create_tool_calling_agent(llm, tools,prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            return agent_executor.stream({"input":user_message})
            """
            llm = ChatOllama(model=model, base_url=base_url)
            # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            # tools = load_tools(["serpapi"], llm)
            # agent_executor = initialize_agent(
            #     tools,
            #     llm=llm,
            #     # agent=AgentType.SELF_ASK_WITH_SEARCH,
            #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            #     verbose=True,
            # )
            res = llm.stream(user_message)

            if isinstance(llm, ChatOllama):
                return (r.content for r in res)
            else:
                return res

        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    Pipe = Pipeline()
    res = Pipe.pipe(
        "What's the date today?",
        "ollama",
        [],
        {},
    )
    for r in res:
        print(r, end="", flush=True)
