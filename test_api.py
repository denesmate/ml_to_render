import json
from fastapi.testclient import TestClient
from api import app


# Instantiate the testing client with our app.
client = TestClient(app)

# Test GET method with "statenumbers/60" endpoint
def test_get_item():
    r = client.get(
        "statenumbers/60",
        headers={"X-Token": "coneofsilence"},
    )
    assert r.status_code == 200
    assert r.json() == 0.6003149606299213

def test_post_data_success():
    data = {"name": "inititial","tags": "example","random_number": 50}
    r = client.post("/statenumbers/", data=json.dumps(data))
    assert r.status_code == 200

def test_post_data_success_result():
    data = {"name": "inititial","tags": "example","random_number": 50}
    r = client.post("/statenumbers/", data=json.dumps(data))
    assert r.json() == {
        "name": "inititial",
        "tags": "example",
        "random_number": 50
    }