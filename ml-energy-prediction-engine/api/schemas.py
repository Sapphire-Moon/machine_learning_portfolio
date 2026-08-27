"""
Pydantic models for the /predict endpoint.

Pydantic is what gives FastAPI its automatic request validation: define
the shape you expect here, and FastAPI rejects anything that doesn't match
before your own code ever runs.
"""
from typing import Literal
from pydantic import BaseModel, Field

Source = Literal["Wind", "Solar"]
DayName = Literal[
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]


class PredictionRequest(BaseModel):
    source: Source = Field(..., description="Energy source: Wind or Solar")
    day_name: DayName = Field(..., description="Day of the week for the target hour")
    start_hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")
    day_of_year: int = Field(..., ge=1, le=366, description="1-366")
    year: int = Field(..., ge=2020, le=2030, description="Calendar year")
    production_lag_1: float = Field(
        ..., ge=0, description="Actual production 1 hour before the target hour"
    )
    production_lag_24: float = Field(
        ..., ge=0, description="Actual production 24 hours before the target hour"
    )
    production_lag_168: float = Field(
        ..., ge=0, description="Actual production 168 hours (1 week) before"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "Wind",
                "day_name": "Monday",
                "start_hour": 14,
                "day_of_year": 200,
                "year": 2025,
                "production_lag_1": 7200,
                "production_lag_24": 6800,
                "production_lag_168": 6500,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: float
    units: str = "energy production units (same scale as the training data)"
