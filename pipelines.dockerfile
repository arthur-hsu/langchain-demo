FROM ghcr.io/open-webui/pipelines:main
RUN pip3 install --no-cache-dir \
    google-search-results       \
    langchain                   \
    langchain-openai            \
    langchain-deepseek          \
    langchain-ollama            \
    langchain-community         \
    langchain-huggingface       \
    langchain-milvus            \
    langchain-text-splitters
RUN huggingface-cli  download BAAI/bge-m3
