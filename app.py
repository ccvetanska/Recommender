import streamlit as st
import pandas as pd
import numpy as np
import torch
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


st.image("assets/movie.png", width=1000)
st.title("Movie Recommender System")
st.sidebar.header("Settings")
selected_model = st.selectbox("Choose model", ["Autoencoder", "Matrix Factorization", "SVD"])
user_id = st.sidebar.selectbox("Choose user", sorted(ratings_df['userId'].unique()))

n = st.sidebar.slider("Recommendations count", 1, 20, 10)

st.subheader(f"Recommendations for user {user_id} with {selected_model}")

if selected_model == "Autoencoder":
    recs = recommend_autoencoder(user_id, model_ae, movie_ids, n)
elif selected_model == "Matrix Factorization":
    recs = recommend_matrix(user_id, model_matrix, n)
# elif selected_model == "BCE":
#     recs = recommend_bce(user_id, model_bce, n)
elif selected_model == "SVD":
    recs = recommend_svd(user_id, svd_model, ratings_df, movies_df, n)
else:
    recs = pd.DataFrame(columns=["movieId", "title"])

st.table(recs.reset_index(drop=True))
