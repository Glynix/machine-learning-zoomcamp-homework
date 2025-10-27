import pickle
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "pipeline_v2.bin"  # this file is inside the base image

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

class Client(BaseModel):
    lead_source: str | None = None
    number_of_courses_viewed: float | int | None = None
    annual_income: float | None = None

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(client: Client):
    record = {
        "lead_source": client.lead_source or "NA",
        "number_of_courses_viewed": float(client.number_of_courses_viewed or 0),
        "annual_income": float(client.annual_income or 0),
    }
    score = float(pipeline.predict_proba([record])[0, 1])
    return {"score": score}
