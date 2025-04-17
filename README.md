# ML Linear Regression Training Example

This repo will walk you through training and validating a Linear Regression ML model that predicts cab fares in the city of Chicago using TensorFlow and Keras. TensorFlow is an open-source machine learning framework developed by Google, primarily used for training and deploying neural networks. It offers a flexible and scalable platform for building and running machine learning models.  Keras is a high-level, user-friendly API for building and training deep learning models.  It was initially developed as an independent library but is now tightly integrated into TensorFlow as its official high-level API. Keras simplifies the process of working with neural networks, making it easier to define, train, and evaluate models.  

Our main focus is on high level machine learning constructs.  This is not intended to be a tutorial on TensorFlow or Keras.

## Linear Regression Basics
Linear Regression is a very common statistical method that allows us to learn the relationship (or function) from a given set of continuous data (training data). For example, we are given some data points **[(X<sub>1</sub>, Y<sub>1</sub>), (X<sub>2</sub>, Y<sub>2</sub>), ... (X<sub>n</sub>, Y<sub>n</sub>)]** and we need to learn the relationship between them or the function **f(X)** that maps a given value of **X** to its corresponding value of **Y**. In the case of Linear regression, the function is a straight line of the form **f(x) = wX + b** where **w** (the weight) and **b** (the bias) are refered to as the parameters of the model. To determine the values of **w** and **b** that best fit the given data points, we will use a **Gradient Descent** optimizer algorithm.

Gradient Descent is an optimization algorithm often used to train machine learning models by locating the minimum values within a loss function. Through an iterative process, gradient descent minimizes the loss function and reduces the difference between predicted and actual results, improving the model’s accuracy. The three types of gradient descent are batch gradient descent, stochastic gradient descent and mini-batch gradient descent. We will be using the mini-batch gradient descent in our example.  This may sound a bit *"MATHY"* right now but walking through the example will help clarify your understanding.
## Setup Environment
We are going to use a conda environment for model training and validation.  Instructions for setting it up can be found below.
```
# If you don't already have conda installed... install miniconda
# The steps outlined below are for a Mac with Apple silicon
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
export PATH=~/miniconda3/condabin:$PATH
conda init
# Don't forget to exit and restart shell

# Create a new conda env
conda create --name keras python=3.11

# Install python dependencies
conda activate keras
conda install keras pandas scikit-learn plotly seaborn tensorflow -y
```
## Basic ML Pipeline
A machine learning (ML) pipeline is a series of interconnected steps that streamline the process of building, training, evaluating, and deploying ML models, from data ingestion to model deployment and monitoring.  We will walk you through the training and validation steps outlined in the basic pipeline depicted below.

<img src="/images/pipeline.png" alt="On Nooo!" witdh="600" height="450">

### Data Analysis, Preparation and Extraction
In an ML pipeline, the data analysis phase, often referred to as data preparation or preprocessing, involves cleaning, transforming, and preparing raw data to make it suitable for model training and analysis, including tasks like exploratory data analysis, feature engineering, and feature selection.

[![something is broken](/images/video-5min.png)](https://www.youtube.com/embed/sMndWXeuFqI "Data Exploration")
### Model Training
Model training is the stage where a machine learning algorithm learns from data to make predictions, involving iterative adjustments of parameters to minimize prediction errors and improve accuracy.  The primary goal of model training is to enable the machine learning model to identify patterns and relationships within the data and make accurate predictions on new, unseen data.

[![something is broken](/images/video-330.png)](https://www.youtube.com/embed/lVncFREcmAI "Training Basics")
[![something is broken](/images/video-620.png)](https://www.youtube.com/embed/qaN1b-h8lF8 "Training Details")
### Model Validation / Evaluation
Model validation within an ML pipeline is the process of evaluating a trained model's performance on unseen data to ensure it generalizes well and meets business objectives. It involves using a separate dataset (validation or test set) that the model hasn't seen during training to assess its predictive capabilities.

[![something is broken](/images/video-600.png)](https://www.youtube.com/embed/fjDF18NBvPY "Model Validation")

⚠️ If you're interested in learning how to serve this model, now that your've trained it... see my [model_serving](https://github.com/sdonovan001/model-serving) repo. :warning:
