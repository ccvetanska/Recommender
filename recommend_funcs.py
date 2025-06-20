import torch
import numpy as np
import pandas as pd

movies_df = pd.read_csv("data/ml-latest-small/movies.csv")
ratings_df = pd.read_csv("data/ml-latest-small/ratings.csv")

unique_user_ids = ratings_df['userId'].unique()
unique_movie_ids = ratings_df['movieId'].unique()
user2idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
movie2idx = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
idx2movie = {v: k for k, v in movie2idx.items()}
user_item_matrix = np.load("data/user_item_matrix.npy")

num_users = len(user2idx)
num_movies = len(movie2idx)

def recommend_autoencoder(user_id, model, movie_ids, n=10):
    user_idx = user2idx[user_id]
    user_input = torch.tensor(user_item_matrix[user_idx]).unsqueeze(0)
    with torch.no_grad():
        preds = model(user_input).numpy().flatten()
    watched = user_item_matrix[user_idx].nonzero()[0]
    rec_indices = [i for i in np.argsort(preds)[::-1] if i not in watched][:n]
    rec_movie_ids = [movie_ids[i] for i in rec_indices]
    return movies_df[movies_df['movieId'].isin(rec_movie_ids)][['movieId', 'title']]

def recommend_matrix(user_id, model, n=10):
    movie_ids = ratings_df['movieId'].unique()
    movie_idxs = [movie2idx[m] for m in movie_ids]
    user_tensor = torch.tensor([user2idx[user_id]] * len(movie_idxs))
    movie_tensor = torch.tensor(movie_idxs)
    with torch.no_grad():
        preds = model(user_tensor, movie_tensor).numpy()
    top_k = np.argsort(preds)[::-1][:n]
    top_ids = [movie_ids[i] for i in top_k]
    return movies_df[movies_df['movieId'].isin(top_ids)][['movieId', 'title']]

def recommend_bce(user_id, model, n=10):
    return recommend_matrix(user_id, model, n)

def recommend_svd(user_id, model, ratings_df, movies_df, n=10):
    watched = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    all_movies = movies_df['movieId'].unique()
    unseen = [m for m in all_movies if m not in watched]

    preds = [(m, model.predict(user_id, m).est) for m in unseen]
    top = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    top_ids = [movie_id for movie_id, _ in top]

    return movies_df[movies_df['movieId'].isin(top_ids)][['movieId', 'title']]

