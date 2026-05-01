# Spotify Mood Clustering + EchoVibe

A Spotify recommendation web app with **two entry points**:

1. **Find Similar Songs** — start from a song you already like and get recommendations using weighted Spotify audio features, from-scratch k-means clustering, weighted squared Euclidean distance, and PCA-based visualization.
2. **Describe a Vibe** — start from a natural-language prompt like *"late night driving"* or *"energetic workout"* and get recommendations using a custom Transformer-based semantic retrieval pipeline with slider-based reranking.

The goal of the app is to make music discovery both **practical** and **interpretable**. A user can either search by a known track or search by a mood, while still keeping direct control over what “similarity” means.

---

## Team

| Member(s) | Role |
|---|---|
| **Shi** | Transformer development, testing (`src/models/transformer.py`, `src/vibe_extension.py`) |
| **Enoch** | Transformer training, testing, data preprocessing (offline training / preprocessing pipeline) |
| **Jonathan** | K-means implementation, transformer integration, testing (`src/kmeans.py`, `src/app.py`) |
| **Gangwon** | Frontend interface, UI/UX, dataset pipeline, app integration (`src/app.py`) |
| **Majo** | Overview and Evaluation |

---

## Project Overview

This app solves two realistic recommendation problems.

### 1. Song-based recommendation
If a user already knows a song they like, the app:
1. represents songs as normalized audio-feature vectors,
2. lets the user assign custom feature weights,
3. clusters songs into broad mood groups using **k-means implemented from scratch in NumPy**,
4. recommends nearby songs using **weighted squared Euclidean distance**,
5. and visualizes the weighted feature space in **2D using PCA**.

### 2. Vibe-based recommendation
If a user does **not** know a specific song but does know the mood they want, the app:
1. encodes the user’s sentence using a custom **Transformer**,
2. retrieves semantically similar songs in a shared latent space,
3. and reranks those candidates using Spotify feature sliders such as energy, danceability, valence, acousticness, speechiness, and tempo.

Together, these two modes make the project both:
- **interpretable** in the classical feature-based setting, and
- **flexible** in the learned semantic-search setting.

---

## Dataset

This project uses the Spotify 1.2M Songs dataset.

**Original Kaggle source:**  
`https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs`

For reproducibility, the large local data files used by this project are provided through the shared Google Drive folder rather than Git.

We intentionally kept several large data artifacts out of version control so the repository would stay lightweight and easier to manage, while still allowing the app to use the larger dataset and embedding files needed for the full two-mode system.

### Files not tracked by Git

The following large files are **not tracked by Git** and must be added locally:

```text
data/tracks_features.csv
data/processed_songs.parquet
data/song_vectors.npy
```

For the full two-mode app, the transformer extension also expects:

```text
models/saved/text_model.pth
```

At the time of writing, `text_model.pth` is already included in the online repository, but the three data files above must still be downloaded separately and placed locally.

---

## Download Instructions

### Recommended setup

Download the required local files from the shared Google Drive folder:

`https://drive.google.com/drive/folders/1jy1K0jz-N8vAYSyWdiRXqWnojHy0bEjs?usp=sharing`

Place them in these exact locations:

```text
data/tracks_features.csv
data/processed_songs.parquet
data/song_vectors.npy
models/saved/text_model.pth
```

### Dataset reference

If needed, the original raw dataset can also be obtained from Kaggle:

`https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs`

However, for this repository, the intended setup is to use the files provided through the shared Drive folder.

---

## About the Dataset

The project uses Spotify audio features to represent songs in a numeric feature space.

### Core song-mode features

The clustering and nearest-neighbor recommendation pipeline uses:
- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo

### Transformer-mode features

The semantic vibe extension uses a 6-feature song representation during training and reranking:
- danceability
- energy
- valence
- acousticness
- speechiness
- tempo

These features are useful because they describe the **acoustic character** of a song rather than its popularity, artist fame, or recommendation history. That makes the app more interpretable and more directly tied to what the music sounds like.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/gswon/CSCI-UA-473-ML-Project.git
cd CSCI-UA-473-ML-Project
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

Your terminal prompt will show `(venv)` when the environment is active.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required data files

Download the required local files from the shared Google Drive folder and place them in the correct locations (see **Download Instructions** above).

### Deactivating the environment

When you are done, run:

```bash
deactivate
```

### Minimum setup for the core app

```text
data/tracks_features.csv
```

### Full setup for the complete two-mode app

```text
data/tracks_features.csv
data/processed_songs.parquet
data/song_vectors.npy
models/saved/text_model.pth
```

---

## Running the App

Make sure the virtual environment is active (`source venv/bin/activate` on macOS/Linux), then from the project root run:

```bash
streamlit run src/app.py
```

The app should launch with four tabs:

- **Find Similar Songs**
- **Describe a Vibe**
- **Evaluation**
- **Project Overview**

If the core dataset is missing, the app will stop and display the expected path.  
If the transformer-extension assets are missing, the song mode will still work, but the vibe mode will warn that the required files are missing.

---

## Running Tests

```bash
pytest tests/
```

---

## Repo Structure

```text
CSCI-UA-473-ML-Project/
├── README.md
├── requirements.txt
├── data/
│   ├── .gitkeep
│   ├── tracks_features.csv        # local dataset file (not tracked by Git)
│   ├── processed_songs.parquet    # local transformer data file (not tracked by Git)
│   └── song_vectors.npy           # local transformer embedding matrix (not tracked by Git)
├── models/
│   └── saved/
│       ├── audio_model.pth
│       └── text_model.pth
├── src/
│   ├── app.py
│   ├── preprocess.py
│   ├── kmeans.py
│   ├── recommend.py
│   ├── reduce.py
│   ├── mood_labels.py
│   ├── vibe_extension.py
│   ├── utils/
│   │   ├── search.py
│   │   ├── cards.py
│   │   ├── df_helpers.py
│   │   ├── thumbnails.py
│   │   └── music_links.py
│   └── models/
│       ├── __init__.py
│       └── transformer.py
├── tests/
│   └── test_kmeans.py
└── training/
```

---

## Algorithm Details

The project contains **two main algorithmic pipelines**.

### 1. Song Representation

In the song-based mode, each song is represented as a numeric feature vector whose coordinates are Spotify audio features such as danceability, energy, loudness, acousticness, speechiness, valence, tempo, and others.

The preprocessing pipeline:
- loads `tracks_features.csv`,
- extracts the continuous audio features,
- drops rows with missing feature values,
- and min-max normalizes the feature matrix so that features live on a comparable scale.

This gives a clean vector space for clustering and recommendation.

### 2. User-Controlled Feature Weighting

The sidebar sliders let the user decide which feature dimensions matter most.

The app applies those weights to the normalized feature matrix before clustering and recommendation. In practice, this means the sliders directly reshape the geometry of the song space.

For example:
- increasing **energy** and **tempo** makes those differences matter more when songs are clustered and ranked,
- increasing **acousticness** emphasizes more acoustic songs,
- setting a feature weight near `0` makes that feature contribute much less to similarity.

### 3. K-Means Clustering (`src/kmeans.py`)

K-means is implemented **from scratch** using NumPy.

The algorithm is based on the standard k-means objective: it tries to minimize the total squared distance from each song to the centroid of the cluster it is assigned to. Equivalently, it tries to make each cluster as compact as possible under squared Euclidean distance.

At a high level, the algorithm repeatedly does two things:

1. **Assignment step:** assign each song to the cluster whose centroid is closest under squared Euclidean distance.
2. **Update step:** replace each centroid with the mean of the songs currently assigned to that cluster.

In other words, the centroid update follows the usual k-means rule that each centroid should be the average of the points in its cluster.

This alternating process continues until the centroids stop moving by more than a small tolerance.

#### Implementation details

The implementation includes:
- **k-means++ initialization**
- **vectorized squared Euclidean distance computation**
- centroid recomputation by cluster means
- convergence based on centroid movement
- inertia tracking for the elbow plot

#### Why k-means is appropriate here

K-means is justified because:
- songs are already represented as vectors in a feature space,
- the course explicitly covered clustering,
- we want broad interpretable “mood neighborhoods,”
- and k-means fits naturally with Euclidean geometry and nearest-neighbor recommendation.

This is one of the most important algorithmic choices in the project.

### 4. Song Recommendation (`src/recommend.py`)

Once a user selects a song:
1. the song is mapped to its normalized feature vector,
2. the current slider weights are applied,
3. the query is assigned to a cluster using the fitted k-means model,
4. nearby songs are retrieved, typically from the same cluster,
5. candidates are ranked using **weighted squared Euclidean distance**.

This ranking works by making feature differences count more heavily when their slider weights are larger. So when a user increases a feature weight, that feature contributes more strongly to the final distance used for recommendation.

This creates a recommendation pipeline that is both **interpretable** and **interactive**.

### 5. PCA Visualization (`src/reduce.py`)

To make clustering interpretable, the app projects the weighted high-dimensional feature space into 2D using PCA.

The PCA implementation:
- centers the weighted data matrix,
- computes its singular value decomposition (SVD),
- and projects the songs onto the first two principal directions.

Conceptually, PCA chooses the two directions that explain the most variance in the weighted feature space, then uses those directions to form a 2D view of the data.

PCA is used **only for visualization**.  
It helps the user:
- inspect the overall structure of the song space,
- see cluster separation,
- and understand how changing slider weights affects the geometry.

It is **not** the actual recommendation metric.

### 6. Choosing the Number of Clusters

The app includes an elbow analysis based on cluster inertia.

Inertia is the k-means objective evaluated on the final clustering: the total within-cluster squared distance. The elbow plot compares that quantity across different values of `k` to help justify the choice of cluster count.

This was included because the course expects algorithmic decisions to be justified, not chosen arbitrarily.

### 7. Song Search Engine (`src/utils/search.py`)

Typing in the search box uses a tiered search pipeline designed to remain responsive on a very large dataset.

The search logic includes:
- **prefix matching** using binary search over sorted normalized song names,
- **substring matching** using a trigram inverted index,
- **fuzzy matching** using a BK-tree with Damerau-Levenshtein distance.

This makes the UI fast enough for a large dataset while still supporting typo tolerance and partial queries.

### 8. Transformer-Based Vibe Search (`src/models/transformer.py`, `src/vibe_extension.py`)

The vibe extension supports natural-language prompts such as:
- `late night driving`
- `energetic workout`
- `calm rainy evening`

The text is encoded by a custom Transformer into a learned sentence embedding. In practical terms, this means the model turns a sentence into a 128-dimensional vector that can be compared to precomputed song vectors.

#### Transformer architecture

The Transformer includes:
- token embeddings,
- positional encoding,
- multi-head self-attention,
- feed-forward layers,
- mean pooling over non-padding tokens,
- and final L2 normalization.

This produces one dense embedding for the entire query sentence.

### 9. Offline Transformer Training and Preprocessing

The transformer extension depends on an **offline pipeline** that was run before the assets were imported into this repository.

The relevant files for that process are:
- `pipeline/reduce_dataset.py`
- `pipeline/train.py`
- `pipeline/index_builder.py`

#### What those steps do

1. **`reduce_dataset.py`** reduces the very large song dataset to a manageable subset for training and experimentation.
2. **`train.py`** trains the text-side Transformer together with the song-side embedding model.
3. **`index_builder.py`** generates the precomputed song embedding matrix used at inference time.

#### What training learns

The semantic pipeline learns a **shared latent space** between:
- text descriptions on the query side,
- and song-feature embeddings on the song side.

Training is based on a contrastive alignment objective plus song-side reconstruction, so that:
- matching text/song pairs are pulled together,
- mismatched pairs are pushed apart,
- and the song embeddings still preserve information about the original song features.

At runtime, this is what allows a sentence prompt to retrieve musically relevant candidates.

### 10. Semantic Retrieval + Slider Reranking (`src/vibe_extension.py`)

The vibe extension uses precomputed local assets:
- `processed_songs.parquet`
- `song_vectors.npy`
- `text_model.pth`

At inference time:
1. the query text is tokenized and embedded,
2. the embedding is compared against the precomputed song vectors,
3. the top semantic candidates are retrieved,
4. those candidates are reranked using user-controlled Spotify feature sliders.

Conceptually, the final ranking is a combination of two ideas:
- semantic similarity between the text prompt and the candidate songs,
- explicit audio-feature preferences provided by the slider settings.

So the final result is not based only on the sentence embedding and not only on the sliders. Instead, the app first finds semantically relevant candidates, then refines those candidates using explicit audio-feature preferences such as energy, danceability, valence, acousticness, speechiness, and tempo.

This makes the vibe mode both:
- **semantic** (matching the meaning of the sentence), and
- **interactive** (letting the user refine the results in real time).

### 11. Why These Methods Fit the Course

This project directly reflects the course themes emphasized in the syllabus:
- **vector representations**
- **similarity metrics**
- **nearest-neighbor retrieval**
- **clustering**
- **dimensionality reduction**
- **transformer embeddings**
- **cross-modal / shared-latent-space retrieval**
- **user-facing ML applications**

The song mode demonstrates a classical vector-space recommendation system.  
The vibe mode extends that with learned semantic retrieval.

---

## Known Limitations

- The song-based mode depends on the quality and consistency of the Spotify audio features.
- PCA is only a visualization tool and does not perfectly preserve geometry.
- The vibe extension depends on offline-generated artifacts being present locally.
- The transformer mode is based on a custom semantic pipeline rather than a large pretrained music-language model.
- The app does not use collaborative filtering or user listening history.

---

## Future Work

Possible future improvements include:
- stronger evaluation of recommendation quality,
- additional UI polish for the vibe mode,
- better balancing of the semantic and slider-based reranking terms,
- broader testing of the semantic model,
- and expanding the text-to-song retrieval pipeline with richer metadata or larger-scale training.