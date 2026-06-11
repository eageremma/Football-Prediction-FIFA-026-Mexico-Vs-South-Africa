"""Train football prediction models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, log_loss


FEATURES = [
    "home_advantage",
    "tournament_importance",
    "home_recent_points",
    "away_recent_points",
    "home_avg_goals_for",
    "away_avg_goals_for",
    "home_avg_goals_against",
    "away_avg_goals_against",
]


def time_split_data(model_data, features=FEATURES, target="result", train_size=0.8):
    """Split data by time order."""
    X = model_data[features]
    y = model_data[target]

    split_index = int(len(model_data) * train_size)

    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def train_logistic_regression(X_train, y_train):
    """Train logistic regression model."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train random forest model."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=20,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print core model evaluation metrics."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print(classification_report(y_test, preds))
    print("Log loss:", log_loss(y_test, probs, labels=model.classes_))
