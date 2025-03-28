To convert the given text into SQL, we need to create a query that processes the data for each IPL season, identifies the top three batsmen and bowlers based on their performance, and then outputs them in the specified format. Here's how you can structure the SQL query:

```sql
WITH SeasonStats AS (
    SELECT 
        season_id,
        player_id,
        role,
        CASE WHEN dismissal_type IN ('run out', 'hit wicket', 'retired hurt') THEN NULL ELSE runs END AS runs,
        CASE WHEN dismissal_type IN ('run out', 'hit wicket', 'retired hurt') THEN NULL ELSE wickets END AS wickets
    FROM 
        matches
),
TopBatsmen AS (
    SELECT 
        season_id,
        player_id,
        SUM(runs) AS total_runs,
        ROW_NUMBER() OVER (PARTITION BY season_id ORDER BY total_runs DESC, player_id ASC) AS rank
    FROM 
        SeasonStats
    WHERE 
        role = 'bat'
    GROUP BY 
        season_id, player_id
),
TopBowlers AS (
    SELECT 
        season_id,
        player_id,
        SUM(wickets) AS total_wickets,
        ROW_NUMBER() OVER (PARTITION BY