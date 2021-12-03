import pandas as pd


def read(path, **kwargs):
    """Return dataframe with relevant columns of CSV file given by path.
    
    Convenience wrapper function for pd.read_csv() with defined subset of
    relevant columns.
    """
    cols = [
        'id', 'home', 'away', 'date', 'year', 'game_status',
        # 'time (utc)', 'attendance', 'venue',
        # 'league', 'part_of_competition', 'shootout',
        'home_score', 'away_score',
        'home_possessionPct', 'away_possessionPct',
        'home_shotsSummary', 'away_shotsSummary',
        'home_foulsCommitted', 'away_foulsCommitted',
        # 'home_yellowCards', 'away_yellowCards',
        # 'home_redCards', 'away_redCards',
        'home_offsides', 'away_offsides',
        'home_wonCorners', 'away_wonCorners',
        'home_saves', 'away_saves',
        # 'home_goal_minutes', 'home_goal_scorers',
        # 'away_goal_minutes', 'away_goal_scorers',
        # 'home_formation', 'away_formation',
        'League'
    ]
    dataframe = pd.read_csv(
        path, low_memory=False, index_col='id', usecols=cols, **kwargs
    )
    return dataframe


if __name__ == "__main__":
    print("Read data")
    df = read('./data/matches.csv')
    print(df.shape)
