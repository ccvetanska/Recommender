import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
from recommend_funcs import recommend_autoencoder, recommend_matrix, recommend_bce, recommend_svd, ratings_df, num_movies, num_users, movies_df
from model_defs import Autoencoder, MatrixFactorizationWithRegularization
from surprise import dump


_, svd_model = dump.load('models/svd_model.pkl')

model_ae = Autoencoder(num_movies)
model_ae.load_state_dict(torch.load("models/best_autoencoder.pt", map_location=torch.device("cpu")))
model_ae.eval()

model_matrix = MatrixFactorizationWithRegularization(num_users, num_movies)
model_matrix.load_state_dict(torch.load("models/model_mfwr.pt", map_location=torch.device("cpu")))
model_matrix.eval()

movie_ids = ratings_df['movieId'].unique()


st.title("Movie Recommender System")

st.sidebar.header("Настройки")
selected_model = st.selectbox("Избери модел", ["Autoencoder", "Matrix Factorization", "BCE", "SVD"])
user_id = st.sidebar.selectbox("Избери потребител", sorted(ratings_df['userId'].unique()))

n = st.sidebar.slider("Брой препоръки", 1, 20, 10)

st.subheader(f"Препоръки за потребител {user_id} чрез {selected_model}")

if selected_model == "Autoencoder":
    recs = recommend_autoencoder(user_id, model_ae, movie_ids, n)
elif selected_model == "Matrix Factorization":
    recs = recommend_matrix(user_id, model_matrix, movie_ids, n)
# elif selected_model == "BCE":
#     recs = recommend_bce(user_id, model_bce, n)
elif selected_model == "SVD":
    recs = recommend_svd(user_id, svd_model, ratings_df, movies_df, n)
else:
    recs = pd.DataFrame(columns=["movieId", "title"])

st.table(recs.reset_index(drop=True))
