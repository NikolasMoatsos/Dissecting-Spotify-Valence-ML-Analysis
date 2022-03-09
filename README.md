# Dissecting-Spotify-Valence-ML-Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NikolasMoatsos/Dissecting-Spotify-Valence-ML-Analysis/blob/main/data_collection.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NikolasMoatsos/Dissecting-Spotify-Valence-ML-Analysis/blob/main/dissecting_spotify_valence_ml_analysis.ipynb)

This project was implemented during the course "Applied Machine Learning", offered by the Department of Management Science & Technology of the Athens University of Economics and Business.

## Purpose
The aim of this project is to untangle the mystery behind Spotify's *valence* and propose how this is derived.

Spotify uses the metric called *valence* to measure the happiness of a track. The metric itself, however, was not developed by Spotify. It was originally developed by Echo Nest, a company that was bought by Spotify in 2014. We don't know exactly how valence is calculated. Some details are given by a blog post, which you can find here:

https://web.archive.org/web/20170422195736/http://blog.echonest.com/post/66097438564/plotting-musics-emotional-valence-1950-2013

This project consists of two notebooks. The first demonstrates the acquisition of the data and the second performs the EDA and the Machine Learning analysis. 

In the analysis notebook the main questions that are attempted to be answered are:
1. Explore which features (track or audio) influence the *valence*, with the use of inferential statistics.
2. Predict the *valence*, with the use of both non-connectivist machine learning methods and neural network methods.

Non-connectivist ML methods used:
* SGD Regressor.
* KNeighbors Regressor.
* Random Forest Regressor.
* XGBoost Regressor.
* LightGBM Regressor.
* Stacking Regressor.

Neural Network methods used:
* Fully Connected Neural Network.
* Convolutional Neural Network.

## Data
The dataset was created with the help of the [Spotify's Web API](https://developer.spotify.com/documentation/web-api/reference/#/) and contains information for both the track and audio features of song. *(More information in the `data_collection.ipynb` notebook)*.

Also, the list of the tracks' ids that were used in the API, was obtained from a dataset on *Kaggle* and can be found [here](https://www.kaggle.com/ektanegi/spotifydata-19212020).

## Technologies
The project was implemented with *Python 3.9.2* and runs on *Jupyter Notebook*. Also includes the following packages: 
* [*pandas*](https://pandas.pydata.org/) *(version used: 1.2.3)* 
* [*NumPy*](https://numpy.org/) *(version used: 1.21.4)*
* [*spotipy*](https://spotipy.readthedocs.io/en/2.19.0/) *(version used: 2.19.0)*
* [*matplotlib*](https://matplotlib.org/) *(version used: 3.3.4)*
* [*seaborn*](https://seaborn.pydata.org/) *(version used: 0.11.2)*
* [*statsmodels*](https://www.statsmodels.org/stable/index.html) *(version used: 0.13.1)*
* [*scikit-learn*](https://scikit-learn.org/stable/) *(version used: 0.24.1)*
* [*XGBoost*](https://xgboost.readthedocs.io/en/stable/) *(version used: 1.5.1)*
* [*LightGBM*](https://lightgbm.readthedocs.io/en/latest/) *(version used: 3.3.1)*
* [*TensorFlow*](https://www.tensorflow.org/) *(version used: 2.7.0)*


