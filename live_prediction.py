import pandas as pd
from ml.prediction import predict_risk
from ml.config import PROCESSED_DATA_PATH
#from ml.data_prep import df
#import data.NER_Landslide_Prototype_Dataset

df = pd.read_csv(PROCESSED_DATA_PATH)

def generate_live_risk_predict(state:str, district:str):
   
   environmental_data=fetch_regional_data(state, district)
   # """generate a landslide risk prediction from current environmental observation."""
   recent= environmental_data.iloc[-1].to_dict()
   result=predict_risk(recent)

   print(type(result))

   return{"predict_type":"live",
          "state":state,
          "district":district,
          "risk_score": result["risk_score"],
          "risk_level": result["risk_level"]
          }

def fetch_regional_data(state:dict, district:dict):
   print(district,state)
   print (f"df:{df}")

   region = df[
      (df["State"] == state)&
      (df["District"] == district)]
   
   # print (f"df:{df}"  )
   print (f"Region:{region}")

   if region.empty:
      raise ValueError(
         f"No data found for {district}, {state}"
      )

   

   return region[
        [
            "State",
            "District",
            "Rainfall_24h_mm",
            "Rainfall_72h_mm",
            "Soil_Moisture_pct",
            "Soil_Depth_cm",
            "Elevation_m",
            "Slope_deg",
            "NDVI",
            "Geology_Risk_Score",
            "Groundwater_m",
            "Earthquake_Magnitude",
            "Previous_Landslide",
        ]
    ]

def generate_ner_alert_status():

    regions = df[
        ["State", "District"]
    ].drop_duplicates()

    results = []

    for _, region in regions.iterrows():

        state = region["State"]
        district = region["District"]

        try:

            result = generate_live_risk_predict(
                state,
                district
            )

            results.append({
                "state": state,
                "district": district,
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"]
            })

        except Exception as e:

            results.append({
                "state": state,
                "district": district,
                "risk_score": None,
                "risk_level": "DATA_UNAVAILABLE",
                "error": str(e)
            })

    return results