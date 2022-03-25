FROM python:3.8-slim

ARG AWS_DEFAULT_REGION
ENV AWS_DEFAULT_REGION="us-east-1"

ARG AWS_DEFAULT_OUTPUT
ENV AWS_DEFAULT_OUTPUT="json"

RUN mkdir /frontend

COPY requirements.txt /frontend

WORKDIR /frontend

RUN pip install -r requirements.txt

RUN pip install awscli --upgrade

COPY . /frontend

EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]