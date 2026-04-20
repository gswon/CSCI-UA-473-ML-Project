from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from models.transformer import TextTransformer


PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed_songs.parquet"
VECTORS_PATH = PROJECT_ROOT / "data" / "song_vectors.npy"
TEXT_MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "text_model.pth"


class CharTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 -,;'\"()[]{}&|!@#$%^*+=_~`.<>?/\\"
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.char2idx["<UNK>"] = len(self.char2idx) + 1
        self.vocab_size = len(self.char2idx) + 1

    def encode(self, text, max_len=64):
        text = str(text).lower()
        indices = [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in text[:max_len]]
        padding = [0] * (max_len - len(indices))
        return indices + padding


@st.cache_resource
def load_vibe_data():
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    return pd.DataFrame(
        columns=[
            "id", "name", "artists", "album",
            "danceability", "energy", "valence",
            "acousticness", "speechiness", "tempo"
        ]
    )


@st.cache_resource
def load_vibe_vectors():
    if VECTORS_PATH.exists():
        return np.load(VECTORS_PATH)
    return None


@st.cache_resource
def load_vibe_transformer():
    tokenizer = CharTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TextTransformer(vocab_size=tokenizer.vocab_size).to(device)

    if TEXT_MODEL_PATH.exists():
        encoder.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
    encoder.eval()
    return encoder, tokenizer, device


def vibe_assets_exist():
    return PARQUET_PATH.exists() and VECTORS_PATH.exists() and TEXT_MODEL_PATH.exists()


def render_vibe_extension(top_n=10):
    st.subheader("Describe the vibe you want")
    st.caption("Experimental extension: semantic text search + slider-based reranking.")

    col1, col2, col3 = st.columns(3)
    with col1:
        slider_energy = st.slider("Vibe Energy", -1.0, 1.0, 0.0, step=0.01, key="vibe_energy")
        slider_dance = st.slider("Danceability", -1.0, 1.0, 0.0, step=0.01, key="vibe_dance")
    with col2:
        slider_mood = st.slider("Mood (Valence)", -1.0, 1.0, 0.0, step=0.01, key="vibe_mood")
        slider_acoustic = st.slider("Acoustic Feel", -1.0, 1.0, 0.0, step=0.01, key="vibe_acoustic")
    with col3:
        slider_speech = st.slider("Vocal Focus (Speechiness)", -1.0, 1.0, 0.0, step=0.01, key="vibe_speech")
        slider_tempo = st.slider("Intensity (Tempo)", -1.0, 1.0, 0.0, step=0.01, key="vibe_tempo")

    query = st.text_input(
        "Describe the vibe you want:",
        placeholder="e.g. Late night driving...",
        key="vibe_query"
    )

    if not query:
        st.info("Enter a sentence above to search by mood, vibe, or listening context.")
        return

    if not vibe_assets_exist():
        st.warning(
            "The transformer extension is missing one or more required files:\n\n"
            f"- {PARQUET_PATH}\n"
            f"- {VECTORS_PATH}\n"
            f"- {TEXT_MODEL_PATH}"
        )
        return

    df = load_vibe_data()
    vectors = load_vibe_vectors()
    encoder, tokenizer, device = load_vibe_transformer()

    if df.empty or vectors is None:
        st.error("Data or vectors are missing. Cannot perform vibe search.")
        return

    with st.spinner("Searching for the perfect vibe..."):
        tokens = tokenizer.encode(query)
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            query_vector = encoder(token_tensor).cpu().numpy().astype(np.float32)
            query_vector = np.ascontiguousarray(query_vector)

        sim_scores = np.dot(vectors, query_vector.T).flatten()

        top_500_idx = np.argsort(sim_scores)[-500:][::-1]
        top_500_scores = sim_scores[top_500_idx]

        if len(top_500_idx) == 0:
            st.warning("No results found.")
            return

        candidate_df = df.iloc[top_500_idx].copy()
        candidate_df["cosine_sim"] = top_500_scores

        feat_arr = candidate_df[
            ["energy", "danceability", "valence", "acousticness", "speechiness", "tempo"]
        ].values

        weights = np.array([
            slider_energy,
            slider_dance,
            slider_mood,
            slider_acoustic,
            slider_speech,
            slider_tempo,
        ])

        shifted_features = feat_arr - 0.5
        slider_contributions = np.sum(shifted_features * weights, axis=1)

        candidate_df["final_score"] = (0.7 * candidate_df["cosine_sim"]) + (0.05 * slider_contributions)

        top_results = candidate_df.sort_values(by="final_score", ascending=False).head(top_n)

    st.markdown(f"### Results for *'{query}'*")
    for _, row in top_results.iterrows():
        st.markdown(
            f"**{row['name']}** by *{row['artists']}* "
            f"(Album: {row['album']}) — Score: `{row['final_score']:.3f}` "
            f"(Semantic sim: `{row['cosine_sim']:.3f}`)"
        )
        cols = st.columns(5)
        cols[0].metric("Energy", f"{row['energy']:.2f}")
        cols[1].metric("Dance", f"{row['danceability']:.2f}")
        cols[2].metric("Valence", f"{row['valence']:.2f}")
        cols[3].metric("Acoustic", f"{row['acousticness']:.2f}")
        cols[4].metric("Tempo", f"{row['tempo']:.2f}")
        st.divider()