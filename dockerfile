FROM python:3.7-slim as builder

RUN apt-get update \
    && apt-get install -y python3-pip locales git libkrb5-dev\
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/

RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt --prefix=/install --no-cache-dir

FROM python:3.7-slim as app
COPY --from=builder /install /usr/local
RUN mkdir road_model
COPY road_model/_road_inference.py road_model/RoadsExtraction_NorthAmerica.emd road_model/RoadsExtraction_NorthAmerica.pth road_model/
COPY run_road_model.py .
ENTRYPOINT ["python3", "-u", "run_road_model.py"]
