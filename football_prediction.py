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


def transform(dataframe, n_trend=5):
    """Return transformed dataframe with features and targets (for regression
    and classification).

    Values are derived from dataframe by calculating the mean of the last
    `n_trend` matches for both home and away team separately and then combining
    those rolling statistics by calculating the difference between each
    "coupled" statistics (e.g. shots_diff = shots_home - shots_away).
    """
    df = dataframe[['home', 'away', 'date', 'home_score', 'away_score']]
    # Add class target
    df = df.assign(winner=_derive_winner(df))
    df = df.sort_values('date')
    stats = _mean_stats_of_last_n_matches(dataframe, n_trend)
    # Add mean stats of home team
    df = df.merge(
        stats, how='left', left_on=['home', 'date'], right_index=True
    )
    # Add mean stats of away team
    df = df.merge(
        stats, how='left', left_on=['away', 'date'], right_index=True,
        suffixes=('_home', '_away')
    )
    # Swap separate mean stats by their respective deltas (difference)
    df = _combine_features(df).set_index(['date', 'home', 'away'])
    return df.dropna()

def _derive_winner(dataframe):
    """Return list with integers indicating who won the matches (0 stands for
    home team, 1 for away team and 2 for a draw).
    """
    goal_diff = dataframe.home_score - dataframe.away_score
    winner = [0 if d>0 else 1 if d<0 else 2 for d in goal_diff]
    return winner

def _mean_stats_of_last_n_matches(dataframe, n=5):
    """Return mean statistics of last n matches for each team."""
    df = _matches_arranged_by_home_and_away_teams(dataframe)
    df = df.drop(columns=['year'])
    club_names = df.reset_index().team.unique()
    all_stats = []
    for club in club_names:
        mean_stats = df.query("team == @club").rolling(
            # After calculating the mean values, shift them backwards by one
            # (otherwise information would leak into "future" target).
            n, min_periods=1).mean().shift()
        all_stats.append(mean_stats)
    return pd.concat(all_stats)

def _matches_arranged_by_home_and_away_teams(dataframe):
    """Return dataframe with matches arranged by teams, i.e. home and away in
    two separate rows instead of one (because we will calculate the mean values
    of the last home *and* away matches for each team).
    """
    home = _matches_arranged_by_home_teams(dataframe)
    away = _matches_arranged_by_away_teams(dataframe)
    df = pd.concat([home, away]).sort_index()
    # Add new column for points
    df = df.assign(points=_derive_points(df))
    return df

def _matches_arranged_by_home_teams(dataframe):
    """Return matches arranged by home team.
    
    Group (sort) dataframe by home and date and rename columns.
    """
    df = dataframe.groupby(['home', 'date']).first(
    ).rename_axis(index={'home': 'team'}
    ).rename(columns={'away': 'opponent',
        'home_score': 'goals', 'away_score': 'goals_opp',
        'home_shots': 'shots', 'away_shots': 'shots_opp',
        'home_foulsCommitted': 'fouls', 'away_foulsCommitted': 'fouls_opp',
        'home_offsides': 'offsides', 'away_offsides': 'offsides_opp',
        'home_wonCorners': 'corners', 'away_wonCorners': 'corners_opp',
        'home_saves': 'saves', 'away_saves': 'saves_opp',
        'home_possession': 'possession', 'away_possession': 'possession_opp',
        }
    )
    return df

def _matches_arranged_by_away_teams(dataframe):
    """Return matches arranged by away team.
    
    Group (sort) dataframe by away and date and rename columns.
    """
    df = dataframe.groupby(['away', 'date']).first(
    ).rename_axis(index={'away': 'team'}
    ).rename(columns={'home': 'opponent',
        'away_score': 'goals', 'home_score': 'goals_opp',
        'away_shots': 'shots', 'home_shots': 'shots_opp',
        'away_foulsCommitted': 'fouls', 'home_foulsCommitted': 'fouls_opp',
        'away_offsides': 'offsides', 'home_offsides': 'offsides_opp',
        'away_wonCorners': 'corners', 'home_wonCorners': 'corners_opp',
        'away_saves': 'saves', 'home_saves': 'saves_opp',
        'away_possession': 'possession', 'home_possession': 'possession_opp',
        }
    )
    return df

def _derive_points(dataframe):
    """Return list with points (3 for victory, 1 for draw, 0 for defeat)."""
    goal_diff = dataframe.goals - dataframe.goals_opp
    points = [3 if diff>0 else 0 if diff<0 else 1 for diff in goal_diff]
    return points

def _combine_features(data):
    """Return dataframe with features combined by calculating the difference
    between each "coupled" columns (e.g. shots_diff = shots_home - shots_away).

    The separate columns (e.g. 'shots_home', 'shots_away') are replaced by the
    delta columns (e.g. 'shots_diff').
    """
    df = data.assign(
        goals_diff = data.goals_home - data.goals_away,
        goals_opp_diff = data.goals_opp_home - data.goals_opp_away,
        shots_diff = data.shots_home - data.shots_away,
        shots_opp_diff = data.shots_opp_home - data.shots_opp_away,
        points_diff = data.points_home - data.points_away,
        possession_diff = data.possession_home - data.possession_away,
        fouls_diff = data.fouls_home - data.fouls_away,
        fouls_opp_diff = data.fouls_opp_home - data.fouls_opp_away,
        offsides_diff = data.offsides_home - data.offsides_away,
        offsides_opp_diff = data.offsides_opp_home - data.offsides_opp_away,
        corners_diff = data.corners_home - data.corners_away,
        corners_opp_diff = data.corners_opp_home - data.corners_opp_away,
        saves_diff = data.saves_home - data.saves_away,
        saves_opp_diff = data.saves_opp_home - data.saves_opp_away
    )
    df = df.drop(columns=[
        'goals_home', 'goals_opp_home',
        'fouls_home', 'fouls_opp_home',
        'offsides_home', 'offsides_opp_home',
        'corners_home', 'corners_opp_home',
        'saves_home', 'saves_opp_home',
        'possession_home', 'possession_opp_home',
        'shots_home', 'shots_opp_home',
        'points_home',
        'goals_away', 'goals_opp_away',
        'fouls_away', 'fouls_opp_away',
        'offsides_away', 'offsides_opp_away',
        'corners_away', 'corners_opp_away',
        'saves_away', 'saves_opp_away',
        'possession_away', 'possession_opp_away',
        'shots_away', 'shots_opp_away',
        'points_away'
    ])
    return df


if __name__ == "__main__":
    print("Read data")
    df = read('./data/matches.csv')
    print("Preprocess data")
    df = preprocess(df)
    print("Transform data")
    df = transform(df)
    print(df.shape)
