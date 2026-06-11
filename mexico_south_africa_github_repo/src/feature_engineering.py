"""Feature engineering functions for football prediction."""

import pandas as pd


def tournament_weight(tournament: str) -> int:
    """Convert tournament type into importance weight."""
    tournament = str(tournament).lower()

    if tournament == "fifa world cup":
        return 5
    if "qualification" in tournament or "qualifier" in tournament:
        return 4
    if "cup" in tournament or "championship" in tournament or "nations league" in tournament:
        return 3
    return 1


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home advantage and tournament importance."""
    df = df.copy()
    df["home_advantage"] = (df["neutral"] == False).astype(int)
    df["tournament_importance"] = df["tournament"].apply(tournament_weight)
    return df


def add_rolling_form_features(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Add rolling recent-form features without leaking future data."""
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

        feature_rows.append(
            {
                "home_recent_points": home_points,
                "home_avg_goals_for": home_gf,
                "home_avg_goals_against": home_ga,
                "away_recent_points": away_points,
                "away_avg_goals_for": away_gf,
                "away_avg_goals_against": away_ga,
            }
        )

        home_score = row["home_score"]
        away_score = row["away_score"]

        home_points_after = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points_after = 3 if away_score > home_score else 1 if home_score == away_score else 0

        team_history.setdefault(home_team, []).append(
            {"points": home_points_after, "goals_for": home_score, "goals_against": away_score}
        )
        team_history.setdefault(away_team, []).append(
            {"points": away_points_after, "goals_for": away_score, "goals_against": home_score}
        )

    return pd.concat([df, pd.DataFrame(feature_rows)], axis=1)


def get_recent_form_for_team(team: str, match_date: str, df: pd.DataFrame, n: int = 10) -> dict:
    """Calculate recent form for a team before a match date."""
    match_date = pd.Timestamp(match_date)

    past_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < match_date)
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
        "avg_goals_against": goals_against / matches_played if matches_played > 0 else 0,
    }
