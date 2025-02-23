FROM ghcr.io/open-webui/pipelines:main
RUN cp ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
