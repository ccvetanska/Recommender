# Movie Recommender System
This is a hybrid movie recommender system built using collaborative filtering, content-based filtering, and autoencoder-based models. The system is deployed through an interactive Streamlit interface and allows two modes of recommendation:

Logged-in user recommendations (based on historical ratings)

Cold-start user input (rate 10 movies and get personalized recommendations)

## Features

* Matrix Factorization (PyTorch) – with regularization, dropout, and early stopping

*  Autoencoder-based collaborative filtering

*  SVD model (using Surprise library)

*  Content-based filtering using TF-IDF on combined metadata

*  Hybrid model that combines content features and personalized user profiles

*  Evaluation Metrics: Precision@k, Recall@k, F1@k (for each model)

*  Streamlit interface with tabbed layout

*  TODO: Binary Classification Approach (BCE Loss)
  
## Models Overview

Content-Based	Content	TF-IDF on genres + description + cast
Matrix Factorization - 	Collaborative	PyTorch model with dropout & early stopping
Autoencoder - Collaborative	Trained to reconstruct the user-item matrix
SVD (Surprise) - Collaborative	Classical model trained with Surprise
Hybrid Content-Based - Content + User-based	Personalized user profiles based on TF-IDF
TODO: BCE-based Classifier - Collaborative	Binary classification on rating threshold

## Evaluation
Metrics are calculated over a test sample of 100 eligible users (≥10 ratings, ≥3 high ratings):

Precision@5 / @10

Recall@5 / @10

F1@5 / @10

Evaluation is consistent across models to ensure fair comparison.

## Streamlit UI
The interface is split into 3 tabs:

User-based Recommendations
Select a known user ID and model to get recommendations.

Rate and Get Recommendations
New users can rate 10 random movies and get cold-start recommendations.

Hybrid Content-Based Tab
Personalized suggestions using a user profile vector and TF-IDF features.

## Requirements

Install dependencies with:
`pip install -r requirements.txt`

#### Keep in mind that you may have problems installing Surprise library. It requires a specific version of C++ to be installed on your machine (14.0)

## How to Run

To start the user interface 
`streamlit run app.py`

## Acknowledgements

This project uses the MovieLens dataset from GroupLens and combines various recommendation strategies for both research and demonstration purposes.
