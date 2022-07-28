import logging
import numpy as np
from geodesic.tesseract.models import serve
from _road_inference import ChildImageClassifier
import torch


class Model:
    def __init__(self):
        self.cls = ChildImageClassifier()
        self.cls.initialize(model="RoadsExtraction_NorthAmerica.emd")

    def inference(self, assets: dict, logger: logging.logger) -> dict:
        array = assets['roads-imagery']

        in_tensor = torch.from_numpy(array)
        output, _ = self.cls.model(in_tensor)

        return {
            'roads': output.numpy()
        }


if __name__ == '__main__':
    model = Model()
    serve(model.inference)
