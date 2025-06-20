import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, num_movies):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_movies)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MatrixFactorizationWithRegularization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, dropout_rate=0.3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, movie_ids):
        user_vec = self.user_emb(user_ids)
        movie_vec = self.movie_emb(movie_ids)
        x = user_vec * movie_vec 
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze()
