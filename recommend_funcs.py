import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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



train_edf = pd.read_csv('data/movies_with_description.csv')
train_edf['description'] = train_edf['description'].fillna('').str.strip()
train_edf['cast'] = train_edf['cast'].fillna('').str.strip()
train_edf['genres'] = train_edf['genres'].fillna('').replace('(no genres listed)', '').str.strip()

train_edf['combined_features'] = (
    train_edf['genres'] + ' ' +
    train_edf['description'] + ' ' +
    train_edf['cast']
)


combo_vectorizer = TfidfVectorizer(stop_words='english')
combo_tfidf_train = combo_vectorizer.fit_transform(train_edf['combined_features'])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(train_edf['combined_features'])

movieId_to_index = pd.Series(train_edf.index, index=train_edf['movieId'])


def build_temp_user_profile(new_rows):
    user_ratings = new_rows
    indices = user_ratings['movieId'].map(movieId_to_index).dropna().astype(int)
    tfidf_vectors = tfidf_matrix[indices]
    ratings = user_ratings.loc[indices.index, 'rating'].values.reshape(-1, 1)
    profile = np.sum(tfidf_vectors.multiply(ratings).toarray(), axis=0)
    return profile / np.linalg.norm(profile)

def recommend_temp_user(n, new_rows, new_user_id, user_ratings_cb):
    profile = build_temp_user_profile(new_rows)
    similarities = cosine_similarity(tfidf_matrix, profile.reshape(1, -1)).flatten()
    watched = list(user_ratings_cb.keys())
    unwatched_mask = ~train_edf['movieId'].isin(watched)
    top_indices = np.argsort(similarities * unwatched_mask.to_numpy())[::-1][:n]
    return train_edf.iloc[top_indices][['title', 'genres', 'description', 'cast']]
