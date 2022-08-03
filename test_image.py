from statistics import mode
from time import sleep
import grpc
from geodesic.tesseract.models import inference_pb2_grpc, inference_pb2
import docker
import argparse
import numpy as np
import pathlib


def get_client(image):
    print("Getting docker client")
    docker_client = docker.from_env()
    print("starting model container")
    model_container = docker_client.containers.run(
        image=image,
        ports={50051: 50051},
        environment={'MODEL_CONTAINER_GRPC_PORT': 50051},
        volumes={
            pathlib.Path().absolute(): {
                'bind': '/data',
                'mode': 'rw'
            }
        },
        detach=True)

    channel = grpc.insecure_channel('localhost:50051')
    print("Waiting for model container server...")
    grpc.channel_ready_future(channel).result(timeout=30)
    print("Connection established.")
    grpc_client = inference_pb2_grpc.InferenceServiceV1Stub(channel)

    # Create some random data and write it to a file on disk as c-order bytes.
    arr = np.random.rand(4, 3, 244, 244)
    bts = arr.astype(np.float32).tobytes()
    with open('test_bytes.arr', 'wb') as fp:
        fp.write(bts)

    return grpc_client, model_container


def run_tests(args):
    grpc_client, model_container = get_client(args.image)

    message = []
    message.append(inference_pb2.SendAssetDataRequest(
        name='roads-imagery',
        type='tensor',
        header=inference_pb2.AssetDataHeader(
            shape=[1, 3, 224, 224],
            dtype='<f32',
            filepath='/data/test_bytes.arr'
        )
    ))
    response = grpc_client.SendAssetData(
        iter(message)
    )

    if response.error:
        print(response.error)

    print(response.name)
    print(response.header.filepath)
    print(response.header.shape)

    print("tests complete, killing contianer.")
    model_container.kill()
    model_container.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    args = parser.parse_args()

    run_tests(args)
