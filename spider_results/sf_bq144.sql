CREATE TABLE ncaa_basketball_tournament_dataset (
    season INT,
    game_outcome VARCHAR(5),
    team_seed INT,
    opponent_seed INT,
    team_name VARCHAR(255),
    opponent_name VARCHAR(255),
    pace_ranking FLOAT,
    efficiency_ranking FLOAT,
    team_stat_value FLOAT,
    opponent_stat_value FLOAT,
    team_opponent_difference FLOAT
);