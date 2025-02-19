FROM ghcr.io/open-webui/pipelines:main
RUN pip3 install --no-cache-dir \
    langchain                   \
    langchain-openai            \
    langchain-deepseek          \
    langchain-ollama            \
    google-search-results       \
    langchain-community
