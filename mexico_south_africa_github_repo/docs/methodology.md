# Methodology

## 1. Problem Definition

The project predicts the 90-minute outcome of Mexico vs South Africa.

Targets:

- Mexico win
- Draw
- South Africa win
- Expected goals
- Most likely scorelines

## 2. Data Collection

The project uses three main data layers:

1. Historical match results
2. Elo ratings
3. Match-specific context

## 3. Data Cleaning

The cleaning process includes:

- Converting match dates to datetime
- Removing missing scores
- Converting scores to integers
- Standardising team names
- Creating result labels
- Creating total goals and goal difference

## 4. Feature Engineering

Features created:

- Home advantage
- Tournament importance
- Recent points
- Average goals scored
- Average goals conceded
- Elo difference
- Match context indicators

## 5. Expected Goals

Expected goals are estimated using team attacking form, opponent defensive weakness, home advantage, Elo difference, and contextual adjustments.

## 6. Poisson Model

The Poisson model estimates scoreline probabilities by treating goals as count-based events.

## 7. Machine Learning Models

Two ML models are trained:

- Logistic Regression
- Random Forest Classifier

The models predict:

- home win
- draw
- away win

## 8. Validation

The models are validated using:

- Accuracy
- Log loss
- Brier score
- Confusion matrix

## 9. Final Blending

Final probabilities are calculated by blending:

- Poisson model
- Elo model
- Logistic Regression
- Random Forest

## 10. Interpretation

The final result is interpreted using both model output and football context.
