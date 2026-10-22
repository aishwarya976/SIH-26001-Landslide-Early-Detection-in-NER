from pydantic import BaseModel, Field


class RiskInput(BaseModel):
    State: str
    District: str

    Rainfall_24h_mm: float
    Rainfall_72h_mm: float

    Soil_Moisture_pct: float
    Soil_Depth_cm: float

    Elevation_m: float
    Slope_deg: float

    NDVI: float
    Geology_Risk_Score: float

    Groundwater_m: float
    Earthquake_Magnitude: float

    Previous_Landslide: int