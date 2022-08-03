import logging
from geodesic.tesseract.models import serve
from road_model._road_inference import ChildImageClassifier
import torch


class Model:
    def __init__(self):
        self.cls = ChildImageClassifier()
        self.cls.initialize(model="road_model/RoadsExtraction_NorthAmerica.emd", model_as_file=True)

    def inference(self, assets: dict, logger: logging.Logger) -> dict:
        array = assets['roads-imagery']

        in_tensor = torch.from_numpy(array).squeeze()
        output, _ = self.cls.model(in_tensor)

        return {
            'roads': output.numpy()
        }


if __name__ == '__main__':
    model = Model()
    serve(model.inference)
