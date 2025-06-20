import streamlit as st
import pandas as pd
import numpy as np
import torch
from surprise import dump

from recommend_funcs import recommend_autoencoder, recommend_matrix, recommend_bce, recommend_svd, ratings_df, num_movies, num_users, movies_df, recommend_temp_user, train_edf
from model_defs import Autoencoder, MatrixFactorizationWithRegularization
from surprise import Dataset, Reader, SVD

from sklearn.metrics.pairwise import cosine_similarity

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

tab1, tab2, tab3 = st.tabs(["Choose User", "Rate & Get Recommendations", "Hybrid Content-Based"])

with tab1:
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

with tab2:
    st.header("Rate 10 random movies and get recommendations with SVD")

    random_movies = movies_df.sample(10, random_state=66).reset_index(drop=True)
    user_ratings = {}

    st.subheader("Please rate the following movies (0 to skip):")

    for i, row in random_movies.iterrows():
        movie_title = row["title"]
        movie_id = row["movieId"]
        rating = st.slider(f"{movie_title}", 0.0, 5.0, 0.0, step=0.5, key=f"rating_{movie_id}")
        if rating > 0:
            user_ratings[movie_id] = rating

    if st.button("Get recommendations based on my ratings"):
        if not user_ratings:
            st.warning("Please rate at least one movie.")
        else:
            new_user_id = ratings_df['userId'].max() + 1
            new_rows = pd.DataFrame({
                "userId": [new_user_id] * len(user_ratings),
                "movieId": list(user_ratings.keys()),
                "rating": list(user_ratings.values())
            })

            temp_ratings = pd.concat([ratings_df, new_rows], ignore_index=True)

            reader = Reader(rating_scale=(0.5, 5.0))
            temp_dataset = Dataset.load_from_df(temp_ratings[['userId', 'movieId', 'rating']], reader)
            trainset = temp_dataset.build_full_trainset()

            temp_svd_model = SVD()
            temp_svd_model.fit(trainset)

            recs_custom = recommend_svd(new_user_id, temp_svd_model, temp_ratings, movies_df, n)
            st.subheader("Recommended for you:")
            st.table(recs_custom.reset_index(drop=True))
with tab3:
    st.header("Hybrid Content-Based Recommendations")
    random_movies = train_edf.sample(10, random_state=42).reset_index(drop=True)
    user_ratings_cb = {}

    st.subheader("Rate the following movies (0 to skip):")

    for i, row in random_movies.iterrows():
        title = row["title"]
        genres = row.get("genres", "Unknown")
        description = row.get("description", "No description available")
        cast = row.get("cast", "No cast info")

        st.markdown(f"**{title}**  \n*Genres:* {genres}  \n*Cast:* {cast}  \n*Description:* {description}")
        rating = st.slider(f"Your rating for '{title}'", 0.0, 5.0, 0.0, step=0.5, key=f"cb_rating_{row['movieId']}")
        if rating > 0:
            user_ratings_cb[row["movieId"]] = rating
        st.markdown("---")

    if st.button("Get content-based recommendations"):
        if not user_ratings_cb:
            st.warning("Please rate at least one movie.")
        else:
            new_user_id = ratings_df['userId'].max() + 1
            new_rows = pd.DataFrame({
                "userId": [new_user_id] * len(user_ratings_cb),
                "movieId": list(user_ratings_cb.keys()),
                "rating": list(user_ratings_cb.values())
            })

            temp_ratings_cb = pd.concat([ratings_df, new_rows], ignore_index=True)

            st.subheader("Recommended for you:")
            st.table(recommend_temp_user(n, new_rows, new_user_id, user_ratings_cb).reset_index(drop=True))