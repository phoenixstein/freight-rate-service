# test_api.py - automatic checks for the API. Run with:  python -m pytest

import pandas as pd                       # reads the December chart file
import pytest                             # the test tool
from fastapi.testclient import TestClient # calls the API without starting a real server

from api import app                       # the application we are testing


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:            # 'with' starts the app, so the model gets loaded
        yield c


def first_december_row():
    """The first row of the December chart file (a load we already predicted and validated)."""
    return pd.read_csv("data/december_chart_inputs.csv").iloc[0]


def body_for(row, **changes):
    """Build the JSON request for a row; keyword arguments override fields."""
    body = {
        "pickup": row["pickup"], "delivery": row["delivery"],
        "distance_miles": float(row["distance"]), "equipment": row["equipment"],
        "weight_lb": float(row["weight"]), "ship_date": str(row["date"])[:10],
    }
    body.update(changes)
    return body


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_prediction_matches_the_validated_december_chart(client):
    row = first_december_row()
    response = client.post("/predict", json=body_for(row))
    assert response.status_code == 200
    data = response.json()
    assert abs(data["predicted_rate"] - float(row["predicted_rate"])) < 0.02   # same as the chart
    assert data["range_low"] < data["predicted_rate"] < data["range_high"]


def test_unknown_city_gives_404(client):
    response = client.post("/predict", json=body_for(first_december_row(), pickup="Atlantis"))
    assert response.status_code == 404


def test_bad_trailer_type_gives_422(client):
    response = client.post("/predict", json=body_for(first_december_row(), equipment="Submarine"))
    assert response.status_code == 422


def test_negative_weight_gives_422(client):
    response = client.post("/predict", json=body_for(first_december_row(), weight_lb=-5))
    assert response.status_code == 422


def test_unknown_date_without_market_data_gives_404(client):
    response = client.post("/predict", json=body_for(first_december_row(), ship_date="2030-01-01"))
    assert response.status_code == 404


def test_unknown_date_works_if_market_data_supplied(client):
    response = client.post("/predict", json=body_for(
        first_december_row(), ship_date="2030-01-01", quote_signal=2.1, market_index=1.0))
    assert response.status_code == 200 and response.json()["notes"]