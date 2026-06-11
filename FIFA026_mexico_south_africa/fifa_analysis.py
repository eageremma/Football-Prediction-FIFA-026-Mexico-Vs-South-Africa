import pandas as pd
import numpy as np

from IPython.display import display

from scipy.stats import poisson

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    brier_score_loss
)

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)


#LOAD DATA
results_path = "C:/Users/User/Desktop/Football Prediction Dataset/results.csv"
elo_path = "C:/Users/User/Desktop/Football Prediction Dataset/Elo_Ratings.xlsx"
context_path = "C:/Users/User/Desktop/Football Prediction Dataset/mexico_south_africa_match_context_dataset.csv"

results = pd.read_csv(results_path)
elo_ratings = pd.read_excel(elo_path)
match_context = pd.read_csv(context_path)

print("Historical results shape:", results.shape)
print("Elo ratings shape:", elo_ratings.shape)
print("Match context shape:", match_context.shape)

display(results.head())
display(elo_ratings.head())
display(match_context.head())

#insepecting the data types and columns

print("Historical results columns:")
print(results.columns.tolist())

print("\nHistorical results data types:")
print(results.dtypes)

print("\nElo columns:")
print(elo_ratings.columns.tolist())

print("\nContext columns:")
print(match_context.columns.tolist())

#cleaning the historical match data

results_clean = results.copy()

# Convert date column into proper datetime format
results_clean["date"] = pd.to_datetime(results_clean["date"], errors="coerce")

# Drop rows that cannot be used for model training
results_clean = results_clean.dropna(
    subset=["date", "home_team", "away_team", "home_score", "away_score"]
).copy()

# Convert scores to integers
results_clean["home_score"] = results_clean["home_score"].astype(int)
results_clean["away_score"] = results_clean["away_score"].astype(int)

# Standardise team names where needed
name_map = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic"
}

results_clean["home_team"] = results_clean["home_team"].replace(name_map)
results_clean["away_team"] = results_clean["away_team"].replace(name_map)

# Create useful score columns
results_clean["total_goals"] = results_clean["home_score"] + results_clean["away_score"]
results_clean["goal_difference"] = results_clean["home_score"] - results_clean["away_score"]

# Create win/draw/loss label
def result_label(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] == row["away_score"]:
        return "draw"
    else:
        return "away_win"

results_clean["result"] = results_clean.apply(result_label, axis=1)

print(results_clean.shape)
display(results_clean.head())
display(results_clean["result"].value_counts(normalize=True).rename("proportion"))


#filtering out for modern football

modern_results = results_clean[results_clean["date"] >= "2000-01-01"].copy()
modern_results = modern_results.sort_values("date").reset_index(drop=True)

modern_results["year"] = modern_results["date"].dt.year
modern_results["recency_weight"] = modern_results["year"] - modern_results["year"].min() + 1

print("Modern dataset shape:", modern_results.shape)
print("Date range:", modern_results["date"].min(), "to", modern_results["date"].max())
display(modern_results.head())

#include the Home advantage

modern_results["home_advantage"] = (modern_results["neutral"] == False).astype(int)

display(modern_results[["home_team", "away_team", "neutral", "home_advantage"]].head())

#including the tournament importance 


def tournament_weight(tournament):
    tournament = str(tournament).lower()

    if tournament == "fifa world cup":
        return 5
    elif "qualification" in tournament or "qualifier" in tournament:
        return 4
    elif "cup" in tournament or "championship" in tournament or "nations league" in tournament:
        return 3
    else:
        return 1

modern_results["tournament_importance"] = modern_results["tournament"].apply(tournament_weight)

display(
    modern_results[["tournament", "tournament_importance"]]
    .drop_duplicates()
    .sort_values("tournament_importance", ascending=False)
    .head(20)
)


#rolling team form


def add_rolling_form_features(df, n=10):
    df = df.sort_values("date").reset_index(drop=True).copy()

    team_history = {}
    feature_rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        def get_team_stats(team):
            history = team_history.get(team, [])[-n:]

            if len(history) == 0:
                return 0, 0.0, 0.0

            points = sum(item["points"] for item in history)
            goals_for = sum(item["goals_for"] for item in history)
            goals_against = sum(item["goals_against"] for item in history)

            return points, goals_for / len(history), goals_against / len(history)

        home_points, home_gf, home_ga = get_team_stats(home_team)
        away_points, away_gf, away_ga = get_team_stats(away_team)

        feature_rows.append({
            "home_recent_points": home_points,
            "home_avg_goals_for": home_gf,
            "home_avg_goals_against": home_ga,
            "away_recent_points": away_points,
            "away_avg_goals_for": away_gf,
            "away_avg_goals_against": away_ga
        })

        # Update team histories only AFTER creating the features for this match
        home_score = row["home_score"]
        away_score = row["away_score"]

        home_points_after = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points_after = 3 if away_score > home_score else 1 if home_score == away_score else 0

        team_history.setdefault(home_team, []).append({
            "points": home_points_after,
            "goals_for": home_score,
            "goals_against": away_score
        })

        team_history.setdefault(away_team, []).append({
            "points": away_points_after,
            "goals_for": away_score,
            "goals_against": home_score
        })

    form_features = pd.DataFrame(feature_rows)
    df = pd.concat([df, form_features], axis=1)

    return df

modern_results = add_rolling_form_features(modern_results, n=10)

form_cols = [
    "home_recent_points",
    "home_avg_goals_for",
    "home_avg_goals_against",
    "away_recent_points",
    "away_avg_goals_for",
    "away_avg_goals_against"
]

display(modern_results[["date", "home_team", "away_team"] + form_cols].tail())

#South Africa and Mexico Recent Form before the Match

match_date = pd.Timestamp("2026-06-11")

def get_recent_form_for_team(team, match_date, df, n=10):
    past_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < match_date)
    ].sort_values("date").tail(n)

    points = 0
    goals_for = 0
    goals_against = 0

    for _, row in past_matches.iterrows():
        if row["home_team"] == team:
            gf = row["home_score"]
            ga = row["away_score"]
        else:
            gf = row["away_score"]
            ga = row["home_score"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            points += 3
        elif gf == ga:
            points += 1

    matches_played = len(past_matches)

    return {
        "team": team,
        "matches_used": matches_played,
        "recent_points": points,
        "avg_goals_for": goals_for / matches_played if matches_played > 0 else 0,
        "avg_goals_against": goals_against / matches_played if matches_played > 0 else 0
    }

mexico_form = get_recent_form_for_team("Mexico", match_date, results_clean, n=10)
south_africa_form = get_recent_form_for_team("South Africa", match_date, results_clean, n=10)
display(pd.DataFrame([mexico_form, south_africa_form]))

#Reading the Elo Ratings, and the Elo ratings difference for the two teams


mexico_elo = int(elo_ratings["mexico_elo"].iloc[0])
south_africa_elo = int(elo_ratings["south_africa_elo"].iloc[0])
elo_diff = mexico_elo - south_africa_elo

print("Mexico Elo:", mexico_elo)
print("South Africa Elo:", south_africa_elo)
print("Elo difference:", elo_diff)

#Inspecting the Match context, to interprete the variable better


context = match_context.iloc[0]

display(match_context.T)

print("Source URL:", context["source_url"])

#estimating the expected goals

home_advantage = int(context["host_nation_advantage"])

mexico_attack = mexico_form["avg_goals_for"]
south_africa_defence = south_africa_form["avg_goals_against"]

south_africa_attack = south_africa_form["avg_goals_for"]
mexico_defence = mexico_form["avg_goals_against"]

mexico_xg = (
    0.45 * mexico_attack +
    0.35 * south_africa_defence +
    0.15 * home_advantage +
    0.05 * (elo_diff / 400)
)

south_africa_xg = (
    0.45 * south_africa_attack +
    0.35 * mexico_defence -
    0.05 * home_advantage -
    0.05 * (elo_diff / 400)
)

# Context adjustment
# Mexico gains from host/crowd/altitude.
# South Africa regains a little because they arrived early and prepared.
mexico_xg += (
    0.05 * int(context["host_nation_advantage"]) +
    0.03 * int(context["home_crowd_advantage"]) +
    0.02 * int(context["altitude_factor"])
)

south_africa_xg += (
    0.03 * int(context["away_team_arrived_early"]) +
    0.02 * int(context["south_africa_preparedness"]) -
    0.02 * int(context["altitude_factor"])
)

# Opening games can be cautious, so slightly reduce both attacks
if int(context["opening_match"]) == 1:
    mexico_xg *= 0.95
    south_africa_xg *= 0.95

mexico_xg = max(mexico_xg, 0.1)
south_africa_xg = max(south_africa_xg, 0.1)

print("Mexico expected goals:", round(mexico_xg, 3))
print("South Africa expected goals:", round(south_africa_xg, 3))

#Using Poisson model for scoreline as football data is a count data


score_probs = {}

max_goals = 7

for mex_goals in range(max_goals + 1):
    for rsa_goals in range(max_goals + 1):
        prob = poisson.pmf(mex_goals, mexico_xg) * poisson.pmf(rsa_goals, south_africa_xg)
        score_probs[(mex_goals, rsa_goals)] = prob

# Normalize because we cut off at max_goals
total_prob = sum(score_probs.values())
score_probs = {score: prob / total_prob for score, prob in score_probs.items()}

top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:10]

top_score_df = pd.DataFrame([
    {
        "scoreline": f"Mexico {score[0]} - {score[1]} South Africa",
        "probability": prob
    }
    for score, prob in top_scores
])

display(top_score_df)

#convert scorelines into win/draws/loss probabilities


poisson_mexico_win = sum(prob for (m, s), prob in score_probs.items() if m > s)
poisson_draw = sum(prob for (m, s), prob in score_probs.items() if m == s)
poisson_south_africa_win = sum(prob for (m, s), prob in score_probs.items() if m < s)

poisson_output = pd.DataFrame([
    {"model": "Poisson", "outcome": "Mexico win", "probability": poisson_mexico_win},
    {"model": "Poisson", "outcome": "Draw", "probability": poisson_draw},
    {"model": "Poisson", "outcome": "South Africa win", "probability": poisson_south_africa_win}
])

display(poisson_output)


#building the machine learning dataset


features = [
    "home_advantage",
    "tournament_importance",
    "home_recent_points",
    "away_recent_points",
    "home_avg_goals_for",
    "away_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_against"
]

model_data = modern_results.dropna(subset=features + ["result"]).copy()

X = model_data[features]
y = model_data["result"]

print("Model data shape:", model_data.shape)
display(X.head())
display(y.value_counts())


#time-based train/test splits


split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Train date range:", model_data.iloc[:split_index]["date"].min(), "to", model_data.iloc[:split_index]["date"].max())
print("Test date range:", model_data.iloc[split_index:]["date"].min(), "to", model_data.iloc[split_index:]["date"].max())


#Train Logistic-Regression baseline


logistic_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"))
])

logistic_model.fit(X_train, y_train)

logistic_probs = logistic_model.predict_proba(X_test)
logistic_preds = logistic_model.predict(X_test)

print(classification_report(y_test, logistic_preds))
print("Logistic log loss:", log_loss(y_test, logistic_probs, labels=logistic_model.classes_))


#train Random Forest model


rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=20,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)
rf_preds = rf_model.predict(X_test)

print(classification_report(y_test, rf_preds))
print("Random Forest log loss:", log_loss(y_test, rf_probs, labels=rf_model.classes_))

print("Confusion matrix:")
display(pd.DataFrame(
    confusion_matrix(y_test, rf_preds, labels=rf_model.classes_),
    index=[f"actual_{c}" for c in rf_model.classes_],
    columns=[f"predicted_{c}" for c in rf_model.classes_]
))


#brier score for home-win probabilities


home_win_true = (y_test == "home_win").astype(int)
home_win_prob = rf_probs[:, list(rf_model.classes_).index("home_win")]

print("Random Forest home-win Brier score:", brier_score_loss(home_win_true, home_win_prob))


#creating the prediction row for mexico and South Africa


mexico_match = pd.DataFrame([{
    "home_advantage": 1,
    "tournament_importance": 5,
    "home_recent_points": mexico_form["recent_points"],
    "away_recent_points": south_africa_form["recent_points"],
    "home_avg_goals_for": mexico_form["avg_goals_for"],
    "away_avg_goals_for": south_africa_form["avg_goals_for"],
    "home_avg_goals_against": mexico_form["avg_goals_against"],
    "away_avg_goals_against": south_africa_form["avg_goals_against"]
}])

display(mexico_match)


#match prediction with machine learning models


rf_match_probs = rf_model.predict_proba(mexico_match)[0]
rf_prediction = dict(zip(rf_model.classes_, rf_match_probs))

logistic_match_probs = logistic_model.predict_proba(mexico_match)[0]
logistic_prediction = dict(zip(logistic_model.classes_, logistic_match_probs))

print("Random Forest prediction:")
print(rf_prediction)

print("\nLogistic Regression prediction:")
print(logistic_prediction)

#building simple Elo model 

# Elo expected score formula
mexico_elo_strength = 1 / (1 + 10 ** (-(elo_diff) / 400))

# Add a realistic draw base for football
elo_draw = 0.25
elo_mexico_win = (1 - elo_draw) * mexico_elo_strength
elo_south_africa_win = (1 - elo_draw) * (1 - mexico_elo_strength)

elo_prediction = {
    "home_win": elo_mexico_win,
    "draw": elo_draw,
    "away_win": elo_south_africa_win
}

print(elo_prediction)


#comparing the model outputs


poisson_prediction = {
    "home_win": poisson_mexico_win,
    "draw": poisson_draw,
    "away_win": poisson_south_africa_win
}

comparison = pd.DataFrame([
    {
        "model": "Poisson score model",
        "Mexico win": poisson_prediction["home_win"],
        "Draw": poisson_prediction["draw"],
        "South Africa win": poisson_prediction["away_win"]
    },
    {
        "model": "Elo model",
        "Mexico win": elo_prediction["home_win"],
        "Draw": elo_prediction["draw"],
        "South Africa win": elo_prediction["away_win"]
    },
    {
        "model": "Logistic Regression",
        "Mexico win": logistic_prediction["home_win"],
        "Draw": logistic_prediction["draw"],
        "South Africa win": logistic_prediction["away_win"]
    },
    {
        "model": "Random Forest",
        "Mexico win": rf_prediction["home_win"],
        "Draw": rf_prediction["draw"],
        "South Africa win": rf_prediction["away_win"]
    }
])

display(comparison)


#blending the models into final probability


weights = {
    "poisson": 0.40,
    "random_forest": 0.30,
    "elo": 0.20,
    "logistic": 0.10
}

final_prediction = {
    "Mexico win": (
        weights["poisson"] * poisson_prediction["home_win"] +
        weights["random_forest"] * rf_prediction["home_win"] +
        weights["elo"] * elo_prediction["home_win"] +
        weights["logistic"] * logistic_prediction["home_win"]
    ),
    "Draw": (
        weights["poisson"] * poisson_prediction["draw"] +
        weights["random_forest"] * rf_prediction["draw"] +
        weights["elo"] * elo_prediction["draw"] +
        weights["logistic"] * logistic_prediction["draw"]
    ),
    "South Africa win": (
        weights["poisson"] * poisson_prediction["away_win"] +
        weights["random_forest"] * rf_prediction["away_win"] +
        weights["elo"] * elo_prediction["away_win"] +
        weights["logistic"] * logistic_prediction["away_win"]
    )
}

# Normalize to ensure probabilities sum to 1
total = sum(final_prediction.values())
final_prediction = {k: v / total for k, v in final_prediction.items()}

final_df = pd.DataFrame([
    {"Outcome": k, "Probability": v, "Percentage": f"{v:.1%}"}
    for k, v in final_prediction.items()
])

display(final_df)
print("Sum of probabilities:", sum(final_prediction.values()))


#final prediction summary

summary = {
    "Mexico expected goals": round(mexico_xg, 2),
    "South Africa expected goals": round(south_africa_xg, 2),
    "Most likely scorelines": top_score_df.head(5)["scoreline"].tolist(),
    "Final Mexico win probability": f"{final_prediction['Mexico win']:.1%}",
    "Final draw probability": f"{final_prediction['Draw']:.1%}",
    "Final South Africa win probability": f"{final_prediction['South Africa win']:.1%}",
}

display(pd.DataFrame([summary]).T.rename(columns={0: "value"}))

