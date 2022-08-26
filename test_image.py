import grpc
from geodesic.tesseract.models import inference_pb2_grpc, inference_pb2
import docker
import argparse
import numpy as np
import pathlib
import os
from geotiff import GeoTiff
from PIL import Image


def get_client(tiff_path, image, size):
    os.makedirs(pathlib.Path().absolute() / "test_output" / "out", exist_ok=True)
    os.makedirs(pathlib.Path().absolute() / "test_output" / "in", exist_ok=True)
    print("Getting docker client")
    docker_client = docker.from_env()
    print("starting model container")
    model_container = docker_client.containers.run(
        image=image,
        ports={50051: 50051},
        environment={'MODEL_CONTAINER_GRPC_PORT': 50051},
        volumes={
            pathlib.Path().absolute() / "test_output": {
                'bind': '/tmp/data',
                'mode': 'rw'
            }
        },
        remove=True,
        detach=True)

    channel = grpc.insecure_channel('localhost:50051')
    print("Waiting for model container server...")
    try:
        grpc.channel_ready_future(channel).result(timeout=30)
    except Exception as e:
        print(model_container.logs().decode())
        raise(e)
    print("Connection established.")
    grpc_client = inference_pb2_grpc.InferenceServiceV1Stub(channel)

    if tiff_path is None:
        arr = np.random.rand(*size)
    else:
        gtiff = GeoTiff(tiff_path, crs_code=3857)
        zarray = gtiff.read()
        arr = np.array(zarray[0:1024, 0:1024, 0:3])
        arr = np.moveaxis(arr, 2, 0)
        arr = np.expand_dims(arr, axis=0)

    bts = arr.astype(np.float32).tobytes()
    with open('test_output/in/test_bytes.arr', 'wb') as fp:
        fp.write(bts)

    return grpc_client, model_container


def run_tests(args):
    input_shape = [1, 3, 1024, 1024]
    grpc_client, model_container = get_client(args.tiff, args.image, input_shape)

    for i in range(1):
        message = []

        for j in range(1):
            message.append(inference_pb2.SendAssetDataRequest(
                name='roads-imagery',
                type='tensor',
                header=inference_pb2.AssetDataHeader(
                    shape=input_shape,
                    dtype='<f4',
                    filepath='/tmp/data/in/test_bytes.arr'
                )
            ))

        try:
            response = grpc_client.SendAssetData(
                iter(message)
            )
        except Exception as e:
            print("exception found. Killing container.")
            print(model_container.logs())
            model_container.kill()
            raise(e)

        resp_iter = iter(response)
        while True:
            try:
                resp = next(resp_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"resp error: {response.code()}")
                print("exception found. Killing container.")
                print(model_container.logs().decode())
                model_container.kill()
                raise(e)
            try:
                print(f"response name: {resp.name}")
                print(f"response filepath: {resp.header.filepath}")
                print(f"response shape: {resp.header.shape}")
                print(f"response dtype: {resp.header.dtype}")

                out_file = resp.header.filepath.split('/')[-1]
                out_arr = np.fromfile("test_output/out/"+out_file, dtype=resp.header.dtype)
                out_arr = np.reshape(out_arr, resp.header.shape)
                out_arr = np.squeeze(out_arr)
                print(f"Array min: {out_arr[ :, :].min()}")
                print(f"Array max: {out_arr[ :, :].max()}")
                print(f"Array mean: {out_arr[ :, :].mean()}")
                print(f"Array std: {out_arr[ :, :].std()}")
                out_arr = out_arr.astype(np.uint8)
                im = Image.fromarray(out_arr[:, :]*255)
                im.save("test_output/out/sample.jpeg")
                os.remove("test_output/out/"+out_file)

            except Exception as e:
                print("exception found. Killing container.")
                print(model_container.logs())
                model_container.kill()
                raise(e)

    print("tests complete, killing contianer.")
    model_container.kill()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--tiff', type=str, default=None)
    args = parser.parse_args()

    run_tests(args)
