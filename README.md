# Football Prediction

Predict scores of football matches:
- home win / away win / draw (=> multiclass classification)
- goals scored by home/away team (=> 2x regression)

## Data
- Source: https://www.kaggle.com/josephvm/european-club-football-dataset?select=matches.csv
- Date:   2021-11-02

## Code
- data.py:       Functions for data preparation
- model.py:      Classes and functions for modeling incl. hyperparameter tuning
- prediction.py: Calls to the above functions to perform multiclass classification and both regression tasks (without hyperparameter tuning)
