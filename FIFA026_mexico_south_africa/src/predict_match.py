"""Run the Mexico vs South Africa prediction pipeline."""

import pandas as pd

from data_preprocessing import load_results, clean_results, filter_modern_football
from feature_engineering import (
    add_basic_features,
    add_rolling_form_features,
    get_recent_form_for_team,
)
from poisson_model import (
    calculate_score_probabilities,
    outcome_probabilities,
    top_scorelines,
)
from train_models import (
    FEATURES,
    time_split_data,
    train_logistic_regression,
    train_random_forest,
)


def main():
    results = load_results("data/raw/results.csv")
    elo_ratings = pd.read_excel("data/raw/Elo_Ratings.xlsx")
    context_data = pd.read_csv("data/raw/mexico_south_africa_match_context_dataset.csv")

    results_clean = clean_results(results)
    modern_results = filter_modern_football(results_clean)
    modern_results = add_basic_features(modern_results)
    modern_results = add_rolling_form_features(modern_results)

    match_date = "2026-06-11"

    mexico_form = get_recent_form_for_team("Mexico", match_date, results_clean)
    south_africa_form = get_recent_form_for_team("South Africa", match_date, results_clean)

    mexico_elo = int(elo_ratings["mexico_elo"].iloc[0])
    south_africa_elo = int(elo_ratings["south_africa_elo"].iloc[0])
    elo_diff = mexico_elo - south_africa_elo

    context = context_data.iloc[0]
    home_advantage = int(context["host_nation_advantage"])

    mexico_xg = (
        0.45 * mexico_form["avg_goals_for"]
        + 0.35 * south_africa_form["avg_goals_against"]
        + 0.15 * home_advantage
        + 0.05 * (elo_diff / 400)
    )

    south_africa_xg = (
        0.45 * south_africa_form["avg_goals_for"]
        + 0.35 * mexico_form["avg_goals_against"]
        - 0.05 * home_advantage
        - 0.05 * (elo_diff / 400)
    )

    mexico_xg += (
        0.05 * int(context["host_nation_advantage"])
        + 0.03 * int(context["home_crowd_advantage"])
        + 0.02 * int(context["altitude_factor"])
    )

    south_africa_xg += (
        0.03 * int(context["away_team_arrived_early"])
        + 0.02 * int(context["south_africa_preparedness"])
        - 0.02 * int(context["altitude_factor"])
    )

    if int(context["opening_match"]) == 1:
        mexico_xg *= 0.95
        south_africa_xg *= 0.95

    mexico_xg = max(mexico_xg, 0.1)
    south_africa_xg = max(south_africa_xg, 0.1)

    score_probs = calculate_score_probabilities(mexico_xg, south_africa_xg)
    poisson_prediction = outcome_probabilities(score_probs)

    model_data = modern_results.dropna(subset=FEATURES + ["result"]).copy()
    X_train, X_test, y_train, y_test = time_split_data(model_data)

    logistic_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    mexico_match = pd.DataFrame(
        [
            {
                "home_advantage": 1,
                "tournament_importance": 5,
                "home_recent_points": mexico_form["recent_points"],
                "away_recent_points": south_africa_form["recent_points"],
                "home_avg_goals_for": mexico_form["avg_goals_for"],
                "away_avg_goals_for": south_africa_form["avg_goals_for"],
                "home_avg_goals_against": mexico_form["avg_goals_against"],
                "away_avg_goals_against": south_africa_form["avg_goals_against"],
            }
        ]
    )

    rf_probs = rf_model.predict_proba(mexico_match)[0]
    rf_prediction = dict(zip(rf_model.classes_, rf_probs))

    logistic_probs = logistic_model.predict_proba(mexico_match)[0]
    logistic_prediction = dict(zip(logistic_model.classes_, logistic_probs))

    mexico_elo_strength = 1 / (1 + 10 ** (-(elo_diff) / 400))

    elo_draw = 0.25
    elo_prediction = {
        "home_win": (1 - elo_draw) * mexico_elo_strength,
        "draw": elo_draw,
        "away_win": (1 - elo_draw) * (1 - mexico_elo_strength),
    }

    weights = {
        "poisson": 0.40,
        "random_forest": 0.30,
        "elo": 0.20,
        "logistic": 0.10,
    }

    final_prediction = {
        "Mexico win": (
            weights["poisson"] * poisson_prediction["home_win"]
            + weights["random_forest"] * rf_prediction["home_win"]
            + weights["elo"] * elo_prediction["home_win"]
            + weights["logistic"] * logistic_prediction["home_win"]
        ),
        "Draw": (
            weights["poisson"] * poisson_prediction["draw"]
            + weights["random_forest"] * rf_prediction["draw"]
            + weights["elo"] * elo_prediction["draw"]
            + weights["logistic"] * logistic_prediction["draw"]
        ),
        "South Africa win": (
            weights["poisson"] * poisson_prediction["away_win"]
            + weights["random_forest"] * rf_prediction["away_win"]
            + weights["elo"] * elo_prediction["away_win"]
            + weights["logistic"] * logistic_prediction["away_win"]
        ),
    }

    total = sum(final_prediction.values())
    final_prediction = {k: v / total for k, v in final_prediction.items()}

    print("Expected Goals")
    print("Mexico:", round(mexico_xg, 2))
    print("South Africa:", round(south_africa_xg, 2))

    print("\nFinal Prediction")
    for outcome, probability in final_prediction.items():
        print(f"{outcome}: {probability:.1%}")

    print("\nMost Likely Scorelines")
    print(top_scorelines(score_probs, "Mexico", "South Africa").head())


if __name__ == "__main__":
    main()
