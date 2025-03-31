import pandas as pd
import wrapper.model as wpr

################# main ####################

validate_df = pd.read_csv("./datasets/validation-data.csv")

features = ["TRIP_MILES"]
label = "FARE"

model = wpr.Model(validate_df, features, label) 
model.load(model_name="fare-model")

model.predict(batch_size=6339)
