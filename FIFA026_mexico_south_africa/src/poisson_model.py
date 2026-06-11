"""Poisson scoreline model."""

from scipy.stats import poisson
import pandas as pd


def calculate_score_probabilities(home_xg: float, away_xg: float, max_goals: int = 7) -> dict:
    """Calculate scoreline probabilities using independent Poisson distributions."""
    score_probs = {}

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
            score_probs[(home_goals, away_goals)] = prob

    total_prob = sum(score_probs.values())
    return {score: prob / total_prob for score, prob in score_probs.items()}


def outcome_probabilities(score_probs: dict) -> dict:
    """Convert scoreline probabilities into win/draw/loss probabilities."""
    return {
        "home_win": sum(prob for (h, a), prob in score_probs.items() if h > a),
        "draw": sum(prob for (h, a), prob in score_probs.items() if h == a),
        "away_win": sum(prob for (h, a), prob in score_probs.items() if h < a),
    }


def top_scorelines(score_probs: dict, home_team: str, away_team: str, n: int = 10) -> pd.DataFrame:
    """Return top scorelines as a DataFrame."""
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:n]

    return pd.DataFrame(
        [
            {
                "scoreline": f"{home_team} {score[0]} - {score[1]} {away_team}",
                "probability": prob,
            }
            for score, prob in top_scores
        ]
    )
