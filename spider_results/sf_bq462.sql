```sql
CREATE TABLE NCAA_Basketball_Records (
    Category VARCHAR(50),
    Date DATE,
    Matchup_or_Venue VARCHAR(255),
    Key_Metric DECIMAL(10, 2)
);

-- Insert data for the first category: Largest venues by seating capacity
INSERT INTO NCAA_Basketball_Records (Category, Date, Matchup_or_Venue, Key_Metric)
SELECT 'Largest Venues by Seating Capacity', 'N/A', venue_name, seating_capacity
FROM venues
WHERE date = 'N/A';

-- Insert data for the second category: National Championship games since the 2016 season with the biggest point margin victories
INSERT INTO NCAA_Basketball_Records (Category, Date, Matchup_or_Venue, Key_Metric)
SELECT 'National Championship Games Since 2016', game_date, matchup, point_margin
FROM championship_games
WHERE game_date >= '2016-01-01'
ORDER BY point_margin DESC
LIMIT 5;

-- Insert data for the third category: Games since the 2011 season with the highest total points scored by both teams combined
INSERT INTO NCAA_Basketball_Records (