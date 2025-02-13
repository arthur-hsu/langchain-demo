FROM ghcr.io/open-webui/pipelines:main
# To Support https://github.com/aporb/webui-extentions/blob/main/deepseek_v3_1.py dependency
RUN pip3 install open_webui --no-cache-dir
