import model_wrapper as wpr
import pandas as pd

################# main ####################

validate_df = pd.read_csv("./datasets/validation-data.csv")

model = wpr.Model() 
model.load("./saved_model/fare-model/1")

#features = ["TRIP_MILES", "TRIP_MINUTES"]
features = ["TRIP_MILES"]
label = "FARE"

model.predict(validate_df, features, label, batch_size=200)
