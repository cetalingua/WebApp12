FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app app/

RUN python3 app/server.py

EXPOSE 8080

CMD ["python3", "app/server.py", "serve"]
