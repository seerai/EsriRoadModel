FROM python:3.8-slim-buster as builder

RUN apt-get update \
    && apt-get install -y python3-pip locales git\
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/

RUN pip3 install --upgrade pip \
    && pip3 install . --user \
    && pip3 install --user -r requirements.txt --no-cache-dir

FROM python:3.7-slim-buster as app
COPY --from=builder /root/.local /root/.local
RUN mkdir road_model
COPY road_model/_road_inference.py road_model/RoadsExtraction_NorthAmerica.emd road_model/RoadsExtraction_NorthAmerica.pth road_model/
COPY run_road_model.py .
ENV PATH=/root/.local/bin/$PATH
CMD ["python3", "-u", "run_road_model.py"]

