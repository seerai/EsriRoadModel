import grp
import grpc
from geodesic.tesseract.models import inference_pb2, inference_pb2_grpc
import docker
import numpy as np
import pathlib
import os
import pytest
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str)
args = parser.parse_args()


@pytest.fixture(scope='session')
def get_client():
    global args
    print(args)
    grpc_client = None
    yield grpc_client

    # teardown
    print("running teardown")


def test_send(get_client):
    assert 1 == 1


if __name__ == '__main__':
    pytest.main(['--show-capture=all', 'pytest_image.py'])
