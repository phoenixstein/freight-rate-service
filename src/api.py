# api.py - a small web service: send a load, get a price prediction and its range.
# Run:  python -m uvicorn --app-dir src api:app --reload     (from the repo root)

from contextlib import asynccontextmanager       # lets us run code at start-up
from datetime import date as Date                # the 'date' type for ship dates
from typing import List, Literal, Optional       # type hints used by Pydantic

import pandas as pd                              # builds the one-row table the model expects
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field            # rules for incoming and outgoing data

from model_store import load_bundle, predict_with_day_median   # our saved model and predictor
from service_data import get_city, get_day_stats               # our SQL lookups


@asynccontextmanager
async def lifespan(app):
    app.state.bundle = load_bundle()             # load the model ONCE when the service starts
    yield                                        # the service now runs until it is stopped


app = FastAPI(title="Freight rate service", lifespan=lifespan)   # the web application object


class PredictRequest(BaseModel):
    """What a caller must send. Pydantic rejects anything that breaks these rules (HTTP 422)."""
    pickup: str                                              # pickup city name
    delivery: str                                            # delivery city name
    distance_miles: float = Field(gt=0, le=4000)             # road miles, must be positive
    equipment: Literal["Dry Van", "Flatbed", "Reefer"]       # only the 3 trailer types we know
    weight_lb: float = Field(gt=0, le=100000)                # pounds, must be positive
    ship_date: Date                                          # e.g. "2025-12-01"
    quote_signal: Optional[float] = Field(default=None, gt=0)    # optional: this load's own quote
    market_index: Optional[float] = Field(default=None, gt=0)    # optional: market level that day


class PredictResponse(BaseModel):
    """What we send back."""
    predicted_rate: float            # dollars
    range_low: float                 # lower end of the typical cushion
    range_high: float                # upper end of the typical cushion
    quote_signal_used: float
    market_index_used: float
    notes: List[str]                 # plain-English warnings, e.g. "used supplied values"


@app.get("/health")
def health(request: Request):
    """A quick 'is the service alive?' check."""
    return {"status": "ok", "model_loaded": request.app.state.bundle is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    """Predict the dollar price of one load."""
    bundle = request.app.state.bundle                        # the model loaded at start-up
    pickup = get_city(req.pickup)                            # (lat, lon) from SQL, or None
    delivery = get_city(req.delivery)
    if pickup is None:
        raise HTTPException(status_code=404, detail=f"Unknown pickup city: {req.pickup}")
    if delivery is None:
        raise HTTPException(status_code=404, detail=f"Unknown delivery city: {req.delivery}")

    notes = []                                               # warnings we will pass back
    day = get_day_stats(req.ship_date.isoformat())           # typical quote and market that day
    if day is None:                                          # we have no stored data for this date
        if req.quote_signal is None or req.market_index is None:
            raise HTTPException(
                status_code=404,
                detail="No stored market data for this date. Send both quote_signal and market_index.",
            )
        day_quote, day_market = req.quote_signal, req.market_index
        notes.append("No stored market data for this date; used the values you supplied.")
    else:
        day_quote, day_market = day
    quote = req.quote_signal if req.quote_signal is not None else day_quote      # caller's value wins
    market = req.market_index if req.market_index is not None else day_market

    row = pd.DataFrame([{                                    # one-row table, same columns as training
        "pickup_lat": pickup[0], "pickup_lon": pickup[1],
        "delivery_lat": delivery[0], "delivery_lon": delivery[1],
        "distance": req.distance_miles, "equipment": req.equipment,
        "weight": req.weight_lb, "date": pd.Timestamp(req.ship_date),
        "market_index": market, "quote_signal": quote,
    }])
    price = float(predict_with_day_median(bundle, row, day_quote)[0])   # run the model
    low, high = bundle["range"]                                         # the calibrated multipliers
    return PredictResponse(
        predicted_rate=round(price, 2),
        range_low=round(price * low, 2),
        range_high=round(price * high, 2),
        quote_signal_used=round(float(quote), 4),
        market_index_used=round(float(market), 4),
        notes=notes,
    )