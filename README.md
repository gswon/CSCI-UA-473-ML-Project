# Spotify Mood Clustering

A Spotify recommendation web app that organizes a large song library into mood-based groupings using **k-means clustering** implemented from scratch, **weighted squared Euclidean distance**, and an interactive **2D visualization** of acoustic feature space.

---

## Team

| Member(s) | Role |
|---|---|
| **Shi & Majo** | Backend algorithm and implementation (`src/kmeans.py`, `src/recommend.py`) |
| **Gangwon** | Frontend interface and UI/UX (`src/app.py`) |
| **Jonathan & Enoch** | Integration, testing, evaluation, and presentation |

---

## Project Overview

Given a Spotify track a user already enjoys, the app:

1. Represents every song as a **feature vector** using Spotify audio features such as tempo, energy, danceability, acousticness, and valence  
2. Lets the user adjust **feature weights** to indicate which audio dimensions matter most  
3. Groups songs into mood clusters using **k-means implemented from scratch in NumPy**  
4. Retrieves the **nearest songs** using **weighted squared Euclidean distance**  
5. Visualizes the feature space in **2D using PCA** so the user can explore how songs relate to one another

The current system is centered on **clustering, nearest-neighbor recommendation, and interactive visualization**. A possible future extension is to add a **transformer-based text-to-track retrieval pipeline** for natural-language mood or vibe prompts.

---

## Dataset

This project uses **`tracks_features.csv`**.

Expected file location:

```text
data/tracks_features.csv
```

The dataset file is **not tracked by Git** because of its size, so each team member should download it locally.

---

## Download Instructions

1. Go to the Kaggle dataset page:  
   `https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs`
2. Download the dataset archive
3. Extract the files
4. Place **`tracks_features.csv`** inside the repository’s `data/` folder

After that, the app should be able to load the dataset directly.

---

## About the Dataset

- The project uses **`tracks_features.csv`**
- The dataset supports:
  - large-scale clustering
  - nearest-neighbor recommendation
  - interactive 2D visualization
- The app expects the dataset to be available locally at `data/tracks_features.csv`

---

## Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/gswon/CSCI-UA-473-ML-Project.git
cd CSCI-UA-473-ML-Project
pip3 install -r requirements.txt
```

If you prefer to use a virtual environment, you may do so, but it is not required.

---

## Running the App

From the project root, run:

```bash
python3 -m streamlit run src/app.py
```

If the dataset is missing, the app will stop and display an error showing the expected file path.

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
├── README.md               # project overview, setup, and usage instructions
├── requirements.txt        # Python dependencies
├── data/
│   ├── .gitkeep            # keeps the folder in Git
│   └── tracks_features.csv # main dataset file (not tracked by Git)
├── notebooks/
│   └── eda.ipynb           # exploratory data analysis
├── src/
│   ├── preprocess.py       # feature extraction, preprocessing, and normalization
│   ├── kmeans.py           # from-scratch k-means implementation
│   ├── recommend.py        # weighted nearest-neighbor recommendation logic
│   ├── reduce.py           # PCA dimensionality reduction to 2D
│   ├── mood_labels.py      # cluster label generation helpers
│   └── app.py              # Streamlit web dashboard
└── tests/
    └── test_kmeans.py      # unit tests for the k-means implementation
```

---

## Algorithm Details

The current recommendation pipeline is based on **weighted audio features, clustering, and nearest-neighbor retrieval**.

### 1. Song Representation
Each song is represented as a vector of Spotify audio features, such as:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo

These features are preprocessed and normalized before clustering and recommendation.

### 2. User-Controlled Feature Weighting
The user can adjust sliders in the Streamlit interface to indicate which features matter more for grouping and recommendation.

Examples:
- increasing **danceability** and **tempo** emphasizes more upbeat songs
- increasing **acousticness** and **valence** emphasizes more mellow or positive songs
- setting a feature weight to **0** removes that feature from consideration

### 3. K-Means Clustering
K-means is implemented **from scratch** using only NumPy.

The clustering procedure follows the standard steps:

- **Initialization:** k-means++ style seeding for more stable starting centroids
- **Assignment:** assign each song to the nearest centroid using **weighted squared Euclidean distance**
- **Update:** recompute each centroid as the mean of the songs assigned to it
- **Stopping:** stop when centroid movement is sufficiently small or a maximum iteration count is reached

### 4. Recommendation
When a user enters a song name:

- the song is mapped to its normalized feature vector
- the current slider weights are applied
- the weighted query is assigned to a cluster
- nearby songs are retrieved, typically from that same cluster
- the results are ranked using **weighted squared Euclidean distance**

Smaller distance means the songs are more similar under the user’s current settings.

### 5. Visualization
To make the clustering interpretable, the app projects the weighted high-dimensional feature space into **2D using PCA**.

This allows the user to:
- see how mood clusters are arranged
- inspect which songs are near each other
- understand how changing slider weights changes the geometry of the song space

### 6. Choosing k
The app also supports an **elbow plot** based on within-cluster inertia.

This helps justify the number of clusters by showing how the clustering objective changes as `k` increases.

---

## Known Limitations

- The current app is based on **audio-feature similarity**, not natural-language semantic search
- Recommendation quality depends on the quality and consistency of the feature columns in `tracks_features.csv`
- PCA is used only for visualization and does **not** define the actual recommendation metric
- Running clustering or elbow analysis on a very large dataset may take noticeable time depending on the machine

---

## Future Work

A possible next step is to support **natural-language mood queries**, such as:

- “late night driving”
- “energetic workout”
- “calm rainy evening”

One possible design is:
1. encode the text query using a transformer
2. retrieve candidate songs in embedding space
3. re-rank those candidates using Spotify audio-feature sliders

That extension is still exploratory and is not required for the current working version of the app.