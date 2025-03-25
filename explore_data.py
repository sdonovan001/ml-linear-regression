import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


dataframe = pd.read_csv("./datasets/taxi-data.csv")
trimmed_df = dataframe[['TRIP_MILES', 'TRIP_MINUTES', 'FARE', 'COMPANY', 'TIP_RATE']]

print("\n------------------------------ Columns of Interest -----------------------------------")
print(trimmed_df.info())

print("\n---------------------------------- Example Data --------------------------------------")
print(trimmed_df.head(200))

print("\n--------------------------------- Data Statistics ------------------------------------")
print(trimmed_df.describe(include='all'))

print("\n--------------------------- Correlation Between Features -----------------------------")
print(trimmed_df.corr(numeric_only=True))

print("\n----------------------------------- Plot Data ----------------------------------------")
sns.pairplot(trimmed_df, x_vars=["TRIP_MILES", "TRIP_MINUTES"], y_vars=["FARE"])
plt.show()

print("\n----------------------- Creating Training / Testing Datasets -------------------------")
training_df, validation_df = train_test_split(trimmed_df, test_size=0.2, random_state=42) 
training_df.to_csv('./datasets/training-data.csv', index=False)
validation_df.to_csv('./datasets/validation-data.csv', index=False)
