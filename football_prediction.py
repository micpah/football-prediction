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


def preprocess(dataframe):
    """Return preprocessed dataframe by calling _clean() and _convert().
    """
    print(dataframe.shape)
    df = _clean(dataframe)
    print(df.shape)
    df = _convert(df)
    print(df.shape)
    return df

def _clean(dataframe):
    """Return cleaned dataframe.
    
    Remove certain rows (e.g. because they are duplicates or have too many 
    missing values ) to assure data quality.
    Remove column 'game_status' (after evaluation).
    """
    df = (dataframe
    .drop_duplicates()  # -10
    # Only keep completed matches
    .query("game_status == 'FT'").drop(columns=['game_status'])  # -6
    # Apparently, statistics were not maintained from the beginning
    .dropna(subset=['home_possessionPct'])  # -2023
    .query("home_possessionPct != '0%' and away_possessionPct != '0%'")  # -683
    .query("not (League == 'English Premier League' and year <= 2002)")  # -379
    .query("not (League == 'German Bundesliga' and year <= 2006)")  # -181
    .query("not (League == 'Dutch Eredivisie' and year <= 2019)")  # -2
    .astype(int, errors='ignore')
    # Rename clubs which have two different names
    .replace('Venezia FC', 'Venezia')
    .replace('Granada CF', 'Granada')
    .replace('Sevilla FC', 'Sevilla')
    )
    return df

def _convert(data):
    """Return dataframe with converted values.
    
    Convert certain values (e.g. string '64%' to float 0.64) and
    swap affected columns.
    """
    # Convert possession percent to real number, e.g '64%' --> 0.64
    h_po = data.home_possessionPct.str.replace('%', '').astype(int) / 100
    a_po = data.away_possessionPct.str.replace('%', '').astype(int) / 100
    # Only take the first number of shotsSummary, e.g. '18 (5)' --> 18
    h_sh = data.home_shotsSummary.str.split('(', expand=True)[0].astype(int)
    a_sh = data.away_shotsSummary.str.split('(', expand=True)[0].astype(int)
    # Add new columns (with converted values)
    df = data.assign(
        date=_derive_date(data),  # Override original date column
        home_possession=h_po, away_possession=a_po,
        home_shots=h_sh, away_shots=a_sh)
    # Drop old columns
    df = df.drop(columns=[
        'home_possessionPct', 'away_possessionPct',
        'home_shotsSummary', 'away_shotsSummary'])
    return df

def _derive_date(dataframe):
    """Return dates (derived by columns 'date' and 'year') with format
    yyyy-mm-dd (e.g. 2021-10-24) in series with dtype datetime64[ns].
    """
    # Extract month and day info from date column
    temp = dataframe.date.str.split(' ', expand=True)
    days = temp[2].astype(int)
    months = pd.to_datetime(temp[1], format='%B').dt.month
    # Original year contains start of season (e.g. 2018 for season 18/19) which
    # is not before August. So we need to add 1 year for the second half of the
    # season (plus summer break), i.e. if date is between January - July.
    years = [y+1 if m<8 else y for y, m in zip(dataframe.year, months)]
    df_date = pd.DataFrame({'year': years, 'month': months, 'day': days})
    return pd.to_datetime(df_date)


if __name__ == "__main__":
    print("Read data")
    df = read('./data/matches.csv')
    print("Preprocess data")
    df = preprocess(df)
