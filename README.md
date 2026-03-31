# 🎵 Spotify Mood Clustering

Automatically organizes a Spotify library into mood-based playlists using **k-means clustering** implemented from scratch, cosine-similarity recommendations, and a 2D visualization of the acoustic feature space.

## Team
| Member | Module |
|---|---|
| Majo & Shi | Algorithm & Implementation (`src/kmeans.py`, `src/recommend.py`) |
| Enoch | Objective & Dataset (`data/`, `notebooks/eda.ipynb`) |
| Jonathan | Significance & Evaluation |
| Gangwon | Dataset pipeline (`src/preprocess.py`) |

## Project Overview

Given a Spotify track a user already enjoys, the app:
1. Represents every song as a **feature vector** (tempo, energy, danceability, etc.)
2. Groups all songs into *k* mood clusters using **k-means** (implemented with NumPy — no scikit-learn)
3. Retrieves the **nearest neighbors** by cosine similarity within the matched cluster
4. **Visualizes** all songs in 2D (via PCA) so the user can see where their taste lands

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) — 32,833 tracks across 6 genres with 12 continuous audio features.

Download `spotify_songs.csv` from Kaggle and place it in `data/`.

This project uses the **Spotify Tracks Dataset** from Kaggle.

## Download Instructions

1. Go to: https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs
2. Click **Download** (free Kaggle account required)
3. Unzip the downloaded file
4. Place `spotify_songs.csv` in this `data/` folder

The CSV is gitignored (too large for GitHub) so every team member needs to download it once.

## About the Dataset

- **32,833 tracks** across 6 playlist genres: pop, rap, rock, latin, R&B, EDM
- **12 continuous audio features** per track: danceability, energy, loudness,
  speechiness, acousticness, instrumentalness, liveness, valence, tempo, and more
- Source: Spotify Web API via Kaggle user @joebeachcapital

## Setup
```bash
git clone https://github.com/gswon/CSCI-UA-473-ML-Project.git
cd CSCI-UA-473-ML-Project
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run src/app.py
```

## Running the EDA Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

## Running Tests
```bash
pytest tests/
```

## Repo Structure
```
CSCI-UA-473-ML-Project/
├── README.md               # this file
├── requirements.txt        # all dependencies
├── data/
│   └── spotify_songs.csv   # Kaggle dataset (not tracked by git)
├── notebooks/
│   └── eda.ipynb           # exploratory data analysis
├── src/
│   ├── preprocess.py       # feature extraction & normalization
│   ├── kmeans.py           # from-scratch k-means clustering (core algorithm)
│   ├── recommend.py        # cosine similarity + neighbor retrieval
│   ├── reduce.py           # PCA dimensionality reduction to 2D
│   └── app.py              # Streamlit web dashboard
└── tests/
    └── test_kmeans.py      # unit tests for k-means implementation
```

## Algorithm Details

K-means is implemented **from scratch** using only NumPy:
- **Initialization**: k++ seeding for better convergence
- **Assignment**: cosine similarity distance metric
- **Update**: recompute centroids as cluster means
- **Stopping**: convergence check on centroid movement + max iterations
```
