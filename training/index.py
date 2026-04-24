# Author: proud cheetah group

import os

# Work around duplicate OpenMP runtime errors that can appear on some local
# PyTorch/macOS setups when generating embeddings.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import pandas as pd
import numpy as np

import sys

# Allow direct execution while still importing the repository-level model class.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.autoencoder import SongAutoencoder


def build_index():
    parquet_path = "data/processed_songs.parquet"
    if not os.path.exists(parquet_path):
        print("Processed data not found. Please run train.py first.")
        return

    print("Loading metadata...")
    df = pd.read_parquet(parquet_path)
    # Must match the feature order used during training, otherwise embeddings
    # will represent the wrong coordinates.
    audio_cols = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "speechiness",
        "tempo",
    ]
    audio_features = df[audio_cols].values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_autoencoder = SongAutoencoder(input_dim=6).to(device)
    model_path = "models/saved/audio_model.pth"
    if os.path.exists(model_path):
        audio_autoencoder.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained audio autoencoder.")
    else:
        print("No trained autoencoder found, using random weights (only for testing!).")

    audio_autoencoder.eval()

    print("Generating embeddings...")
    batch_size = 1000
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(audio_features), batch_size):
            # Encode in batches to keep memory usage predictable on laptops and
            # machines without a large GPU.
            batch = torch.tensor(
                audio_features[i : i + batch_size], dtype=torch.float32
            ).to(device)
            embeds = audio_autoencoder.encode(batch)
            all_embeddings.append(embeds.cpu().numpy())

            if (i // batch_size) % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(audio_features)} songs...")

    embeddings = np.ascontiguousarray(np.vstack(all_embeddings).astype("float32"))

    # Pure Python/NumPy L2 normalization keeps saved vectors ready for cosine
    # similarity without requiring FAISS at index-build time.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    index_path = "data/song_vectors.npy"
    np.save(index_path, embeddings)
    print(
        f"Vectors successfully saved to {index_path} with {len(embeddings)} songs (FAISS bypassed!)."
    )


if __name__ == "__main__":
    build_index()
