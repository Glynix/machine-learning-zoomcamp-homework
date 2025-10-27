import pickle
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "pipeline_v1.bin"

# Load once at startup
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

class Client(BaseModel):
    lead_source: str | None = None
    number_of_courses_viewed: int | float | None = None
    annual_income: float | None = None

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(client: Client):
    # DictVectorizer in the pipeline expects a plain dict
    record = {
        "lead_source": client.lead_source if client.lead_source is not None else "NA",
        "number_of_courses_viewed": float(client.number_of_courses_viewed or 0),
        "annual_income": float(client.annual_income or 0.0),
    }
    # Pipeline returns probabilities
    score = float(pipeline.predict_proba([record])[0, 1])
    return {"score": score}
