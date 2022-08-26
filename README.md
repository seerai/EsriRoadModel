# Getting DLPK working

Using the road model for North America which can be found 
[here](https://seerai.maps.arcgis.com/home/item.html?id=0c00be3c7e4042ebadd3ae1404190a5b).

There is also a global model but it uses TensorFlow so I don’t want to deal with trying to get it running.

In the DLPK there are two scripts `ArcGISImageClassifier.py` and `_road_inference.py`. From what I can tell the first one is just
a wrapper for the second one and since we just want access to the model itself we can just use `_road_inference.py`.

You can initialize and get access to the model with:
```python
from arcgis.learn.models._multi_task_road_extractor import MultiTaskRoadExtractor

road_extractor = MultiTaskRoadExtractor.from_model(
    data=None,
    emd_path="road_model/RoadsExtraction_NorthAmerica.emd"
)
model = road_extractor.learn.model
model.eval()
output_tensor, _ = model(input_tensor)
```

### Build the Container
The dockerfile is in the root directory. You can build it with:
```bash
docker build -t roadmodel:v0.0.1 -f dockerfile .
```

### Test Locally
There is a script that will test to make sure the docker container can properly communicate with tesseract. To run the test:
```bash
python test_image.py --image roadmodel:v0.0.1
```


## NOTE:
This repo does not contain the `pth` file with the network weights. You must download and unzip the dlpk from the link above
and put it into the `road_model` folder for the docker image to build.