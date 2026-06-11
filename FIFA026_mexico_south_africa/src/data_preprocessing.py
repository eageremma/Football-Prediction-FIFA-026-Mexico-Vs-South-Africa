"""Data preprocessing functions for football prediction."""

import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Load historical football results."""
    return pd.read_csv(path)


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
    """Clean historical football match results."""
    df = results.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    name_map = {
        "USA": "United States",
        "Korea Republic": "South Korea",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
    }

    df["home_team"] = df["home_team"].replace(name_map)
    df["away_team"] = df["away_team"].replace(name_map)

    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_difference"] = df["home_score"] - df["away_score"]

    def result_label(row):
        if row["home_score"] > row["away_score"]:
            return "home_win"
        if row["home_score"] == row["away_score"]:
            return "draw"
        return "away_win"

    df["result"] = df.apply(result_label, axis=1)

    return df


def filter_modern_football(results: pd.DataFrame, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Keep only modern football matches."""
    df = results[results["date"] >= start_date].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["recency_weight"] = df["year"] - df["year"].min() + 1
    return df
