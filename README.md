# Getting DLPK working

Using the road model for North America which can be found 
[here](https://seerai.maps.arcgis.com/home/item.html?id=0c00be3c7e4042ebadd3ae1404190a5b).

There is also a global model but it uses TensorFlow so I don’t want to deal with trying to get it running.

You can initialize and get access to the model with something like:
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
The Geodesic CLI can test to make sure the docker container can properly communicate with tesseract. To run the test:
```bash
geodesic validate roadmodel:v0.0.1
```


## NOTE:
This repo does not contain the `pth` file with the network weights. You must download and unzip the dlpk from the link above
and put it into the `road_model` folder for the docker image to build.