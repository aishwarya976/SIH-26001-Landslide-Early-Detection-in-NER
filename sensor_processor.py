from datetime import datetime


def process_sensor_data(
    sensor_data: dict,
) -> dict:

    processed_data = {
        "sensor_id": sensor_data.get("sensor_id"),

        "timestamp": sensor_data.get("timestamp",
            datetime.utcnow().isoformat(),
        ),

        "latitude": sensor_data.get("latitude"),

        "longitude": sensor_data.get("longitude"),

        "soil_moisture": sensor_data.get("soil_moisture"),

        "pore_pressure": sensor_data.get("pore_pressure"),

        "ground_movement": sensor_data.get("ground_movement"),

        "tilt": sensor_data.get("tilt"),

        "rainfall": sensor_data.get("rainfall"),
    }

    return processed_data