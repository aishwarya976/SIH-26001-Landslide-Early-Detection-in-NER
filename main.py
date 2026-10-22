from fastapi import FastAPI

from api.routes.risk import router as risk_router


app = FastAPI(
    title="NER Landslide Risk Monitoring System",
    description=(
        "AI-based early warning and landslide "
        "risk monitoring system for the "
        "North Eastern Region of India."
    ),
    version="1.0.0",
)


app.include_router(risk_router)


@app.get("/")
def root():

    return {
        "message": "NER Landslide Risk Monitoring System"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }