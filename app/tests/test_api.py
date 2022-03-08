import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from ..app import app


@pytest.fixture
def client():
    # use "with" statement to run "startup" event of FastAPI
    with TestClient(app) as c:
        yield c


def test_app_prediction(client):
    """ Test prediction response"""
    filename = 'app/tests/ovw_dog.jpg'

    response = client.post("/prediction",
                           files={"raw_image": ("filename", open(
                               filename, "rb"), "image/jpeg")}
                           )

    try:
        assert response.status_code == 200
        reponse_json = response.json()
        print(response.json())
        assert reponse_json['category'] == "overweight"

    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise
