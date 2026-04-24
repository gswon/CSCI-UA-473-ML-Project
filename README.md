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
| **Majo** | Presentation and evaluation |

---

## Project Overview

This app solves two realistic recommendation problems.

### 1. Song-based recommendation
If a user already knows a song they like, the app:
1. represents songs as normalized audio-feature vectors,
2. lets the user assign custom feature weights,
3. clusters songs into broad mood groups using **k-means implemented from scratch in NumPy**,
4. recommends nearby songs using **weighted squared Euclidean distance**, and
5. visualizes the weighted feature space in **2D using PCA**.

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

This project uses the Spotify 12M Songs dataset.

**Original Kaggle source:**  
`https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs`

For reproducibility, the large local data files used by this project are provided through the shared Google Drive folder rather than Git.

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

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/gswon/CSCI-UA-473-ML-Project.git
cd CSCI-UA-473-ML-Project
pip3 install -r requirements.txt
```

A virtual environment is optional but recommended.

After installing dependencies, download the required local files from the shared Drive folder and place them in the correct folders.

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

From the project root, run:

```bash
python3 -m streamlit run src/app.py
```

The app should launch with four tabs:

- **Find Similar Songs**
- **Describe a Vibe**
- **Evaluation**
- **Project Overview**

If the core dataset is missing, the app will stop and display the expected path.  
If the transformer-extension assets are missing, the song mode will still work, but the vibe mode will warn that the required files are missing.

---

## Running the EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

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
│       └── text_model.pth         # local transformer weights file
├── notebooks/
│   └── eda.ipynb
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
└── tests/
    └── test_kmeans.py
```

---

## Algorithm Details

The project contains **two main algorithmic pipelines**.

---

### 1. Song Representation

In the song-based mode, each song is represented as a numeric feature vector

\[
x_i \in \mathbb{R}^d
\]

where the coordinates are Spotify audio features such as danceability, energy, loudness, acousticness, speechiness, valence, tempo, and others.

The preprocessing pipeline:
- loads `tracks_features.csv`,
- extracts the continuous audio features,
- drops rows with missing feature values,
- and min-max normalizes the feature matrix so that features live on a comparable scale.

This gives a clean vector space for clustering and recommendation.

---

### 2. User-Controlled Feature Weighting

The sidebar sliders let the user decide which feature dimensions matter most.

If

\[
w = (w_1, \dots, w_d)
\]

is the user’s feature-weight vector, then the app applies those weights to the normalized feature matrix before clustering and recommendation.

This means the sliders are not cosmetic — they directly change the geometry of the song space.  
For example:
- increasing **energy** and **tempo** emphasizes more energetic/upbeat tracks,
- increasing **acousticness** emphasizes more acoustic songs,
- setting a feature weight near `0` makes that feature largely irrelevant.

---

### 3. K-Means Clustering (`src/kmeans.py`)

K-means is implemented **from scratch** using NumPy.

The algorithm minimizes within-cluster squared distance:

\[
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2
\]

where:
- \(C_k\) is cluster \(k\),
- \(\mu_k\) is the centroid of cluster \(k\).

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

---

### 4. Song Recommendation (`src/recommend.py`)

Once a user selects a song:
1. the song is mapped to its normalized feature vector,
2. the current slider weights are applied,
3. the query is assigned to a cluster using the fitted k-means model,
4. nearby songs are retrieved, typically from the same cluster,
5. candidates are ranked using **weighted squared Euclidean distance**.

A clean form of the distance is:

\[
d(q, x_i) = \sum_{j=1}^{d} w_j (q_j - x_{i,j})^2
\]

Smaller distance means more similar under the current user settings.

This creates a recommendation pipeline that is both **interpretable** and **interactive**.

---

### 5. PCA Visualization (`src/reduce.py`)

To make clustering interpretable, the app projects the weighted high-dimensional feature space into 2D using PCA.

PCA is used **only for visualization**.  
It helps the user:
- inspect the overall structure of the song space,
- see cluster separation,
- and understand how changing slider weights affects the geometry.

It is **not** the actual recommendation metric.

---

### 6. Choosing the Number of Clusters

The app includes an elbow analysis based on cluster inertia.

This helps justify the choice of \(k\) by showing how the clustering objective changes as \(k\) increases.

This was included because the course expects algorithmic decisions to be justified, not chosen arbitrarily.

---

### 7. Song Search Engine (`src/utils/search.py`)

Typing in the search box uses a tiered search pipeline designed to remain responsive on a very large dataset.

The search logic includes:
- **prefix matching** using binary search over sorted normalized song names,
- **substring matching** using a trigram inverted index,
- **fuzzy matching** using a BK-tree with Damerau-Levenshtein distance.

This makes the UI fast enough for a large dataset while still supporting typo tolerance and partial queries.

---

### 8. Transformer-Based Vibe Search (`src/models/transformer.py`, `src/vibe_extension.py`)

The vibe extension supports natural-language prompts such as:

- `late night driving`
- `energetic workout`
- `calm rainy evening`

The text is encoded by a custom Transformer:

\[
q = T(\text{text})
\]

where \(q \in \mathbb{R}^{128}\) is a learned sentence embedding.

#### Transformer architecture
The Transformer includes:
- token embeddings,
- positional encoding,
- multi-head self-attention,
- feed-forward layers,
- mean pooling over non-padding tokens,
- and final L2 normalization.

This produces one dense embedding for the entire query sentence.

---

### 9. Offline Transformer Training and Preprocessing

The transformer extension depends on an **offline pipeline** that was run before the assets were imported into this repository.

The relevant files for that process are:

- `pipeline/reduce_dataset.py`
- `pipeline/train.py`
- `pipeline/index_builder.py`

#### What those steps do
1. **`reduce_dataset.py`**  
   reduces the very large song dataset to a manageable subset for training and experimentation.

2. **`train.py`**  
   trains the text-side Transformer together with the song-side embedding model.

3. **`index_builder.py`**  
   generates the precomputed song embedding matrix used at inference time.

#### What training learns
The semantic pipeline learns a **shared latent space** between:
- text descriptions on the query side,
- and song-feature embeddings on the song side.

Training is based on a contrastive alignment objective plus song-side reconstruction, so that:
- matching text/song pairs are pulled together,
- mismatched pairs are pushed apart,
- and the song embeddings still preserve information about the original song features.

At runtime, this is what allows a sentence prompt to retrieve musically relevant candidates.

---

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

A simplified final score looks like:

\[
\text{FinalScore}_i
=
\lambda \,\text{sim}(q, v_i)
+
\alpha \,\langle f_i, w \rangle
\]

where:
- \(q\) is the text-query embedding,
- \(v_i\) is the candidate song embedding,
- \(f_i\) is the candidate’s audio-feature vector,
- \(w\) is the slider-weight vector.

This makes the vibe mode both:
- **semantic** (matching the meaning of the sentence), and
- **interactive** (letting the user refine the results in real time).

---

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