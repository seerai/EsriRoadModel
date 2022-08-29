import logging
from geodesic.tesseract.models import serve
from arcgis.learn.models._multi_task_road_extractor import MultiTaskRoadExtractor
import torch
import numpy as np
import time


class Model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emd_path = "road_model/RoadsExtraction_NorthAmerica.emd"
        self.road_extractor = MultiTaskRoadExtractor.from_model(
            data=None,
            emd_path=self.emd_path
        )
        self.model = self.road_extractor.learn.model.to(self.device)
        self.model.eval()

    def inference(self, assets: dict, logger: logging.Logger) -> dict:
        logger.info("Got inference request")
        array = assets['roads-imagery']
        array = array.copy()

        logger.info("running esri road model")
        in_tensor = torch.from_numpy(array).to(self.device)
        in_tensor = in_tensor[:, 0:3, :, :]  # Get the first 3 bands. The 4th is just a mask.
        output, _ = self.model(in_tensor)

        output = output.softmax(dim=1)[:, [1], :, :].gt(0.5)
        output = output.cpu().detach().numpy()

        return {
            'roads': output
        }

    def get_model_info(self):
        return {
            'inputs': [
                {
                    'name': 'road_model',
                    'dtype': '<f4',
                    'shape': [1, 4, 1024, 1024]
                }
            ],
            'outputs': [
                {
                    'name': 'roads',
                    'dtype': '|u1',
                    'shape': [1, 1, 1024, 1024]
                }
            ]
        }


if __name__ == '__main__':
    model = Model()
    serve(model.inference)
