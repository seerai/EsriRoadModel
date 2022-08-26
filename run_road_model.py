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
        # Normalize the array. These values are based on the transforms found in the EMD file.
        in_tensor = in_tensor[:, 0:3, :, :]  # Get the first 3 bands. The 4th is just a mask.
        output, _ = self.model(in_tensor)

        # output = apply_hanning_filter(output, output.size[2], device=self.device)

        output = output.softmax(dim=1)[:, [1], :, :]
        output = output.gt(0.5)
        output = output.cpu().detach().numpy()

        # Do the shape padding to get 1k x 1k guaranteed output. This will change soon in tesseract. Need it for now.
        output_padded = np.zeros((1, 1, 1024, 1024), dtype=np.uint8)
        output_padded[:output.shape[0], :output.shape[1], :output.shape[2], :output.shape[3]] = output[:]

        return {
            'roads': output_padded
        }

    def get_model_info(self):
        return {}

    def apply_hanning_filter(input, window_size, device):
        """Apply the hanning filter to the image chip.

        Apply a hanning filter so that predictions are weighted up in the middle of the chip compared to the outside.
        """
        hann_window = torch.hann_window(window_size, device=device).unsqueeze(0)
        window = hann_window * hann_window.T
        output = input * window
        return output


if __name__ == '__main__':
    model = Model()
    serve(model.inference)
