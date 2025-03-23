# TF / Keras ML Linear Regression Example
---
## Overview
TensorFlow is a widely used, open-source machine learning framework developed by Google.  It is primarily used for training and deploying neural networks. Tensorflow offers a flexible and scalable platform for building and running machine learning models.  Keras is a high-level, user-friendly API for building and training deep learning models.  It was initially developed as an independent library but is now tightly integrated into TensorFlow as its official high-level API. Keras simplifies the process of working with neural networks, making it easier to define, train, and evaluate models.

This repo will walk you through using Tendorflow and Keras to create and serve a Linear Regression ML model. The model will predict cab fares in the city of chicago using TRIP_MILES and TRIP_MINUTES as inputs.
## Linear Regression Basics
Linear Regression is a very common statistical method that allows us to learn a relationship (or function) from a given set of continuous data (training data). For example, we are given some data points of x and corresponding y and we need to learn the relationship between them or the **function** that maps a given value of x to its corresponding value of y. In the case of Linear regression, the function is a straight line, i.e, **f(x) = wX + b**. The **weight** (w) and the **bias** (b) are refered to as the parameters of the model. 
## Example Data
| TRIP_MILES | TRIP_MINUTES | FARE | COMPANY | TIP_RATE |
|------------|--------------|------|---------|----------|
| 2.57 | 39.016667 | 31.99 | Flash Cab | 6.3 |
| 1.18 | 17.900000 | 9.75  | Flash Cab | 27.9 |
| 1.29 | 19.550000 | 10.25 | Sun Taxi | 0.0 |
| 3.70 | 56.000000 | 23.75 | Choice Taxi Association | 0.0 |
| 1.15 | 17.400000 | 10.00 | Flash Cab | 0.0 |
| ...  | ... | ... | ... | ... |

## Setup conda Environment
```
# If you don't already have conda installed... install miniconda
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

