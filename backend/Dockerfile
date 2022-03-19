## For CPU version uncomment next RUN statement
# FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

# ENV CONDA_AUTO_UPDATE_CONDA=false \
#     PATH=/opt/miniconda/bin:$PATH



# RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
#     && chmod +x ~/miniconda.sh \
#     && ~/miniconda.sh -b -p /opt/miniconda \
#     && rm ~/miniconda.sh \
#     && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

## For GPU version uncomment the next RUN statement
# RUN pip3 --timeout=300 --no-cache-dir install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip3 --timeout=300 --no-cache-dir install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html


COPY ./requirements.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt

# Copy model files
COPY ./model /model

# Copy app files
COPY ./app /app
WORKDIR /app/
ENV PYTHONPATH=/app
RUN ls -lah /app/*

# COPY ./start.sh /start.sh
# RUN chmod +x /start.sh

EXPOSE 8080
# CMD ["/start.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]