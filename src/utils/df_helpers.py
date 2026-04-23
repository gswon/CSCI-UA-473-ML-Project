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
