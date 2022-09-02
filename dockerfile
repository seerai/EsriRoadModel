FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04 as builder

RUN apt-get update \
    && apt-get install -y software-properties-common

RUN apt-add-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3-pip python3.7 python3.7-dev locales libkrb5-dev\
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/
RUN python3.7 -m pip install --upgrade pip \
    && python3.7 -m pip install -r requirements.txt --prefix=/install

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04 as app
RUN apt-get update \ 
    && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.7 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
COPY --from=builder /install /usr/local
RUN mkdir road_model
COPY road_model/RoadsExtraction_NorthAmerica.emd road_model/RoadsExtraction_NorthAmerica.pth road_model/
COPY run_road_model.py .
ENV PYTHONPATH=/usr/local/lib/python3.7/site-packages
CMD ["python3.7", "-u", "run_road_model.py"]

