import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder for system call sequences.
    """
    def __init__(self, sequence_length, num_system_calls, embedding_dim, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim  # Store embedding_dim

        self.embedding = nn.Embedding(num_system_calls, embedding_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(sequence_length * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), encoding_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length * embedding_dim)  # Output size matches the total size of embedded input
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, self.sequence_length)  # Reshape to [batch_size, sequence_length, embedding_dim]
        return decoded
