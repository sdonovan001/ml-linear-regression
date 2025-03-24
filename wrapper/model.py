import os
os.environ['WRAPT_DISABLE_EXTENSIONS'] = "true"
import pandas as pd
import numpy as np

import keras
import tensorflow as tf
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Model:
   def __init__(self, learning_rate=0.001, epochs=20, batch_size=50):
      self._learning_rate = learning_rate
      self._epochs = epochs
      self._batch_size = batch_size

   def train(self, training_data, feature_names, label_name):
      self._num_features = len(feature_names) 
      features = training_data.loc[:, feature_names].values
      label = training_data[label_name].values 
      self._training_data = training_data
      self._feature_names = feature_names
      self._label_name = label_name
 
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

      # Gather the trained model's weight and bias.
      self._trained_weight = self._model.get_weights()[0]
      self._trained_bias = self._model.get_weights()[1]

      # Isolate the error for each epoch.
      self._rmse = pd.DataFrame(history.history)["root_mean_squared_error"]
      self._epochs = history.epoch
   
      # To track the progression of training, we're going to take a snapshot
      # of the model's root mean squared error at each epoch.
      self._equation = "{} = {:.3f} * {}".format(label_name, self._trained_weight[0][0], feature_names[0])
      self._equation += " + {:.3f}".format(self._trained_bias[0])

      print("=====================================================")
      print("Model Fit Equation: {}".format(self._equation))

   def set_params(self, learning_rate=0.001, epochs=20, batch_size=50):
      self._learning_rate = learning_rate
      self._epochs = epochs
      self._batch_size = batch_size

   def graph(self, sample_size=200):
      random_sample = self._training_data.sample(n=sample_size).copy()
      random_sample.reset_index()

      is_2d_plot = len(self._feature_names) == 1
      model_plot_type = "scatter" if is_2d_plot else "surface"
      model_title = "Model Plot<br>" + self._equation
      last_rmse = self._rmse.iloc[-1]
      loss_title = "Loss Curve<br>RMSE: {:.4f}".format(last_rmse)
      fig = make_subplots(rows=1, cols=2,
                          subplot_titles=(loss_title, model_title),
                          specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

      self._plot_data(random_sample, self._feature_names, self._label_name, fig)
      self._plot_model(random_sample, self._feature_names, self._trained_weight, self._trained_bias, fig)
      self._plot_loss_curve(self._epochs, self._rmse, fig)

      hyper_params = "Hyperparameters (Learning Rate: {} Epocs: {} Batch Size: {})".format(self._learning_rate, len(self._epochs), self._batch_size)
      fig.update_layout(title_text=hyper_params, title_x=0.5)
      fig.show()
      return

   def save(self, path):
      self._model.save(path)
      print("Model saved to {}".format(path))

   def load(self, path):
      self._model = tf.keras.saving.load_model(path)
      print("Model loaded from {}".format(path))
      self._print_banner("MODEL SUMMARY")
      print(self._model.summary())

   def _print_banner(self, banner_text):
      print("\n")
      header = "-" * 80
      banner = header + "\n" + "|" + banner_text.center(78) + "|" + "\n" + header
      print(banner)

   def predict(self, df, features, label, batch_size=50):
      batch = df.sample(n=batch_size).copy()
      batch.set_index(np.arange(batch_size), inplace=True)

      self._print_banner("EVALUATE MODEL")
      self._model.evaluate(x=batch.loc[:, features].values, y=batch.loc[:, label].values)
      predicted_values = self._model.predict(x=batch.loc[:, features].values)

      data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [], features[0]: []}

      def format_currency(x):
         return "${:.2f}".format(x)

      for i in range(batch_size):
         predicted = predicted_values[i][0]
         observed = batch.at[i, label]
         data["PREDICTED_FARE"].append(format_currency(predicted))
         data["OBSERVED_FARE"].append(format_currency(observed))
         data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
         data[features[0]].append(batch.at[i, features[0]])

      output_df = pd.DataFrame(data)

      self._print_banner("PREDICTIONS")
      print(output_df)

   def _plot_data(self, df, features, label, fig):
      if len(features) == 1:
         scatter = px.scatter(df, x=features[0], y=label)
      else:
         scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

      fig.append_trace(scatter.data[0], row=1, col=2)
      if len(features) == 1:
         fig.update_xaxes(title_text=features[0], row=1, col=2)
         fig.update_yaxes(title_text=label, row=1, col=2)
      else:
         fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

      return
 
   def _plot_loss_curve(self, epochs, rmse, fig):
      curve = px.line(x=epochs, y=rmse)
      curve.update_traces(line_color='#ff0000', line_width=3)
   
      fig.append_trace(curve.data[0], row=1, col=1)
      fig.update_xaxes(title_text="Epoch", row=1, col=1)
      fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

      return

   def _plot_model(self, df, features, weights, bias, fig):
      df['FARE_PREDICTED'] = bias[0]

      for index, feature in enumerate(features):
         df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

      if len(features) == 1:
         model = px.line(df, x=features[0], y='FARE_PREDICTED')
         model.update_traces(line_color='#ff0000', line_width=3)
      else:
         z_name, y_name = "FARE_PREDICTED", features[1]
         z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
         y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
         x = []
         for i in range(len(y)):
            x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

         plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

         light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
         model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                           colorscale=light_yellow))

      fig.add_trace(model.data[0], row=1, col=2)

      return
