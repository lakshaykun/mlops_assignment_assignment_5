from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert "current_model" in data
    assert data["status"] == "running"


def test_model_info():
    response = client.get("/info")
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert isinstance(data, dict)


def test_predict_valid():
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], int)


def test_predict_invalid_feature_length():
    payload = {
        "features": [5.1, 3.5]  # invalid input
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data