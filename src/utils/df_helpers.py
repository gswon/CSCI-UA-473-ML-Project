"""
Small DataFrame column-detection helpers.

The two Spotify datasets we support use different column names
(`name`/`artists` vs `track_name`/`track_artist`). These helpers let
callers pick the right one without repeating the conditional.
"""

import pandas as pd


def get_name_col(df: pd.DataFrame) -> str:
    return "name" if "name" in df.columns else "track_name"


def get_artist_col(df: pd.DataFrame) -> str:
    return "artists" if "artists" in df.columns else "track_artist"


# Single source of truth for cleaning the raw `artists` field, which
# arrives as a Python list-repr string like "['Foo', 'Bar']" in the 1.2M
# dataset. Used everywhere a normalized artist or dedup key is needed,
# so dedup keys, cache keys, and search URLs stay in sync.

def clean_artist(artist) -> str:
    return str(artist).replace("[", "").replace("]", "").replace("'", "").strip()


def clean_artist_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()


def dedup_key(name, artist) -> str:
    return f"{str(name).strip().lower()}||{clean_artist(artist).lower()}"


def dedup_key_series(name_s: pd.Series, artist_s: pd.Series) -> pd.Series:
    return (
        name_s.astype(str).str.strip().str.lower()
        + "||"
        + clean_artist_series(artist_s).str.lower()
    )
