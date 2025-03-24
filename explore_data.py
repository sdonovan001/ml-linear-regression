import io
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("./datasets/taxi-data.csv")
trimmed_ds = dataset[['TRIP_MILES', 'TRIP_MINUTES', 'FARE', 'COMPANY', 'TIP_RATE']]

print("\n------------------------------ Columns of Interest -----------------------------------")
print(trimmed_ds.info())

print("\n---------------------------------- Example Data --------------------------------------")
print(trimmed_ds.head(200))

print("\n--------------------------------- Data Statistics ------------------------------------")
print(trimmed_ds.describe(include='all'))

print("\n--------------------------- Correlation Between Features -----------------------------")
print(trimmed_ds.corr(numeric_only=True))

sns.pairplot(trimmed_ds, x_vars=["TRIP_MILES", "TRIP_MINUTES"], y_vars=["FARE"])
plt.show()

print("\n----------------------- Creating Training / Testing Datasets -------------------------")
training_ds, validation_ds = train_test_split(trimmed_ds, test_size=0.2, random_state=42) 
training_ds.to_csv('./datasets/training-data.csv', index=False)
validation_ds.to_csv('./datasets/validation-data.csv', index=False)
