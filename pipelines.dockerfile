FROM ghcr.io/open-webui/pipelines:main
RUN apt update && apt install -y vim
RUN cp ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
