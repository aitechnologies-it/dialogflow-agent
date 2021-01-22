# Container image that runs your code
FROM python:3.8.6-alpine

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY . /

RUN pip3 install -r /requirements.txt
RUN chmod +x /entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]