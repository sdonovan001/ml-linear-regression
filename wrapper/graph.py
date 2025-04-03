import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Graph:
   def __init__(self, model):
      self._model = model 

   def show(self, sample_size=200):
      random_sample = self._model._data.sample(n=sample_size).copy()
      random_sample.reset_index()

      is_2d_plot = len(self._model._feature_names) == 1
      model_plot_type = "scatter" if is_2d_plot else "surface"
      model_title = "Model Plot<br>" + self._model._equation

      fig = None

      if self._model._trained:
         last_rmse = self._model._rmse.iloc[-1]
         loss_title = "Loss Curve<br>RMSE: {:.4f}".format(last_rmse)
         fig = make_subplots(rows=1, cols=2,
                             subplot_titles=(loss_title, model_title),
                             specs=[[{"type": "scatter"}, {"type": model_plot_type}]])
         self._plot_loss_curve(self._model._epochs, self._model._rmse, fig)
         hyper_params = "Hyperparameters (Learning Rate: {} Epochs: {} Batch Size: {})".format(self._model._learning_rate,
                                                                                              len(self._model._epochs),
                                                                                              self._model._batch_size)
         fig.update_layout(title_text=hyper_params, title_x=0.5)
      else:
         model_title = "Model Plot (RMSE: {:.4f})<br>{}".format(self._model._rmse, self._model._equation)
         fig = make_subplots(rows=1, cols=1,
                             subplot_titles=(model_title,),
                             specs=[[{"type": model_plot_type}]])

      self._plot_data(random_sample, self._model._feature_names, self._model._label_name, fig)
      self._plot_model(random_sample, self._model._feature_names, self._model._trained_weight, self._model._trained_bias, fig)

      fig.show()

   def _plot_data(self, df, features, label, fig):
      col = 1
      if self._model._trained:
         col = 2

      if len(features) == 1:
         scatter = px.scatter(df, x=features[0], y=label)
      else:
         scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

      fig.append_trace(scatter.data[0], row=1, col=col)
      if len(features) == 1:
         fig.update_xaxes(title_text=features[0], row=1, col=col)
         fig.update_yaxes(title_text=label, row=1, col=col)
      else:
         fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

   def _plot_loss_curve(self, epochs, rmse, fig):
      curve = px.line(x=epochs, y=rmse)
      curve.update_traces(line_color='#ff0000', line_width=3)
   
      fig.append_trace(curve.data[0], row=1, col=1)
      fig.update_xaxes(title_text="Epoch", row=1, col=1)
      fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

   def _plot_model(self, df, features, weights, bias, fig):
      col = 1
      if self._model._trained:
         col = 2

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

      fig.add_trace(model.data[0], row=1, col=col)
