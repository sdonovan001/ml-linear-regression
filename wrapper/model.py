import os
os.environ['WRAPT_DISABLE_EXTENSIONS'] = "true"
import pandas as pd
import numpy as np

import keras
import tensorflow as tf

from wrapper import graph

class Model:
   def __init__(self, data, feature_names, label_name, learning_rate=0.001, batch_size=50, epochs=20):
      self._num_features = len(feature_names)
      self._data = data
      self._feature_names = feature_names
      self._label_name = label_name

      self._learning_rate = learning_rate
      self._batch_size = batch_size
      self._epochs = epochs

      self._trained = False

   def train(self):
      self._trained = True
      features = self._data.loc[:, self._feature_names].values
      label = self._data[self._label_name].values 
 
      self._print_banner("TRAINING MODEL")

      inputs = keras.Input(shape=(self._num_features,))
      outputs = keras.layers.Dense(units=1)(inputs)
      self._model = keras.Model(inputs=inputs, outputs=outputs)

      self._model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=self._learning_rate),
                          loss="mean_squared_error",
                          metrics=[keras.metrics.RootMeanSquaredError()])
      history = self._model.fit(x=features,
                                y=label,
                                batch_size=self._batch_size,
                                epochs=self._epochs)

      # Save off trained model's weight and bias for later use.
      self._trained_weight = self._model.get_weights()[0]
      self._trained_bias = self._model.get_weights()[1]

      # Isolate the error for each epoch.
      self._rmse = pd.DataFrame(history.history)["root_mean_squared_error"]
      self._epochs = history.epoch
   
      self._set_equation()

      print("=====================================================")
      print("Model Fit Equation: {}".format(self._equation))

   def predict(self, batch_size=50):
      batch = self._data.sample(n=batch_size).copy()
      batch.set_index(np.arange(batch_size), inplace=True)

      self._print_banner("EVALUATE MODEL")
      self._model.evaluate(x=batch.loc[:, self._feature_names].values, y=batch.loc[:, self._label_name].values)
      predicted_values = self._model.predict(x=batch.loc[:, self._feature_names].values)

      data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [], self._feature_names[0]: []}

      def format_currency(x):
         return "${:.2f}".format(x)

      for i in range(batch_size):
         predicted = predicted_values[i][0]
         observed = batch.at[i, self._label_name]
         #data["PREDICTED_FARE"].append(format_currency(predicted))
         #data["OBSERVED_FARE"].append(format_currency(observed))
         data["PREDICTED_FARE"].append(predicted)
         data["OBSERVED_FARE"].append(observed)
         data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
         data[self._feature_names[0]].append(batch.at[i, self._feature_names[0]])

      output_df = pd.DataFrame(data)

      self._print_banner("PREDICTIONS")
      print(output_df)

      self._trained_weight = self._model.get_weights()[0]
      self._trained_bias = self._model.get_weights()[1]
      self._set_equation()

   def set_params(self, learning_rate=0.001, batch_size=50, epochs=20):
      self._learning_rate = learning_rate
      self._batch_size = batch_size
      self._epochs = epochs

   def graph(self, sample_size=200):
      self._graph = graph.Graph(self) 
      self._graph.show(sample_size)

   def save(self, model_name="default"):
      path = "./saved_models/{}/1".format(model_name)
      self._model.save(path)
      print("Model saved to {}".format(path))

   def load(self, model_name="default"):
      path = "./saved_models/{}/1".format(model_name)
      self._model = tf.keras.saving.load_model(path)
      print("Model loaded from {}".format(path))
      self._print_banner("MODEL SUMMARY")
      print(self._model.summary())

   def _print_banner(self, banner_text):
      print("\n")
      header = "-" * 80
      banner = header + "\n" + "|" + banner_text.center(78) + "|" + "\n" + header
      print(banner)

   def _set_equation(self):
      if len(self._feature_names) == 1:
         self._equation = "{} = {:.3f} * {}".format(self._label_name, self._trained_weight[0][0], self._feature_names[0])
         self._equation += " + {:.3f}".format(self._trained_bias[0])
      else:
         self._equation = "{} = {:.3f} * {}".format(self._label_name, self._trained_weight[0][0], self._feature_names[0])
         self._equation += " + {:.3f} * {}".format(self._trained_weight[1][0], self._feature_names[1])
         self._equation += " + {:.3f}".format(self._trained_bias[0])
