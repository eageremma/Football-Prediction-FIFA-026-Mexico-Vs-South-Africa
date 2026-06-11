# Football-Prediction-FIFA-026-Mexico-Vs-South-Africa
# Mexico vs South Africa FIFA World Cup 2026 Prediction Analysis

![Project Cover](presentation/sports_analytics_cover.png)

## Project Overview

This project uses Python and data science methods to analyse and predict the FIFA World Cup 2026 opening match between **Mexico** and **South Africa**.

The aim is not to guarantee a match result, but to estimate football outcomes using probability-based modelling.

The project estimates:

- Mexico win probability
- Draw probability
- South Africa win probability
- Expected goals for both teams
- Most likely scorelines
- Model comparison between Poisson, Elo, Logistic Regression, and Random Forest

## Match Analysed

| Item | Detail |
|---|---|
| Match | Mexico vs South Africa |
| Competition | FIFA World Cup 2026 |
| Date | 11 June 2026 |
| Venue | Mexico City |
| Analysis Type | Football prediction / Sports analytics |
| Tools | Python, Pandas, Scikit-learn, SciPy, Jupyter Notebook |

## Project Structure

```text
mexico_south_africa_github_repo/
│
├── data/
│   ├── raw/
│   │   ├── results.csv
│   │   ├── Elo_Ratings.xlsx
│   │   └── mexico_south_africa_match_context_dataset.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── mexico_south_africa_prediction_notebook.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── poisson_model.py
│   ├── train_models.py
│   └── predict_match.py
│
├── outputs/
│   ├── figures/
│   └── predictions/
│
├── presentation/
│   ├── mexico_south_africa_linkedin_carousel.pptx
│   └── sports_analytics_cover.png
│
├── docs/
│   ├── methodology.md
│   └── data_sources.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Used

### 1. Historical International Match Results

The main training dataset contains historical international football matches with columns such as:

- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`
- `tournament`
- `city`
- `country`
- `neutral`

This dataset is used to train the model on past international football outcomes.

### 2. Elo Ratings

Elo ratings were used as a team-strength indicator.

For this analysis:

| Team | Elo Rating |
|---|---:|
| Mexico | 1875 |
| South Africa | 1517 |

### 3. Match Context Dataset

The match-context dataset includes:

- Mexico as host nation
- Mexico City altitude factor
- Opening-match pressure
- South Africa early arrival/preparation
- Home-crowd advantage
- Tactical context from reliable news reporting

## Methodology

The analysis follows these steps:

1. Define the prediction problem
2. Load raw datasets
3. Clean historical match data
4. Standardise team names
5. Filter to modern football from 2000 onwards
6. Engineer recent-form features
7. Add home advantage
8. Add tournament-importance weighting
9. Add Elo strength difference
10. Estimate expected goals
11. Use Poisson model for scorelines
12. Train Logistic Regression and Random Forest models
13. Validate the model using log loss, Brier score, and confusion matrix
14. Create Mexico vs South Africa prediction row
15. Compare model outputs
16. Blend final probabilities
17. Interpret the result with football context

## Models Used

| Model | Purpose |
|---|---|
| Poisson Model | Estimate scoreline probabilities |
| Elo Model | Estimate team-strength advantage |
| Logistic Regression | Simple machine-learning baseline |
| Random Forest | Non-linear match-outcome classifier |
| Blended Model | Final balanced prediction |

## Example Final Output

| Outcome | Probability |
|---|---:|
| Mexico win | ~49–50% |
| Draw | ~32–33% |
| South Africa win | ~18–19% |

Expected goals:

| Team | Expected Goals |
|---|---:|
| Mexico | ~1.30 |
| South Africa | ~0.71 |

Most likely scorelines:

- Mexico 1–0 South Africa
- Mexico 1–1 South Africa
- Mexico 2–0 South Africa
- Mexico 2–1 South Africa

## Key Interpretation

Mexico are slight favourites due to:

- Home advantage
- Stronger Elo rating
- Mexico City conditions
- Crowd support
- Host-nation advantage

However, South Africa still have a meaningful chance because:

- They prepared early for Mexico City altitude
- Opening matches are often tight
- Football is low-scoring and uncertain
- A draw remains very realistic

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mexico-south-africa-worldcup-prediction.git
cd mexico-south-africa-worldcup-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook notebooks/mexico_south_africa_prediction_notebook.ipynb
```

Or run:

```bash
jupyter lab
```

## Requirements

Main Python libraries:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- openpyxl
- jupyter

## Limitations

This model has limitations:

- Football is low-scoring and random.
- Lineups can change close to kick-off.
- Free data sources may be incomplete.
- Historical results do not fully capture current squad quality.
- The model does not include player-level event data.
- Opening World Cup matches can behave differently from normal matches.
- This is an analytical project, not betting advice.

## Portfolio Value

This project demonstrates:

- Data collection
- Data cleaning
- Feature engineering
- Predictive modelling
- Probability modelling
- Sports analytics
- Machine learning validation
- Communication of analytical findings
- LinkedIn-ready storytelling

## Author

**Paul Emmanuel**  
Data Science / Machine Learning / Sports Analytics Project
