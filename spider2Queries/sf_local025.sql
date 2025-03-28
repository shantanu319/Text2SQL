To achieve the desired result using SQL, you need to perform several steps. Here's a step-by-step breakdown of how you can construct the query:

1. **Combine Runs**: Calculate the total runs scored in each over for both batsmen.
2. **Identify Highest Over**: Find the over with the highest total runs.
3. **Retrieve Bowler**: Get the bowler for that over from the ball by ball table.
4. **Calculate Average**: Compute the average of these highest over totals across all matches.

Assuming you have two tables: `matches` and `ball_by_ball`. The `matches` table contains information about each match, and the `ball_by_ball` table contains detailed information about each ball played in a match.

Here's a possible SQL query to accomplish this:

```sql
WITH CombinedRuns AS (
    SELECT 
        m.match_id,
        b.over,
        SUM(CASE WHEN b.batsman = 'bat_1' THEN b.runs ELSE 0 END) + SUM(CASE WHEN b.batsman = 'bat_2' THEN b.runs ELSE 0 END) AS total_runs,
        SUM(b.extra_runs) AS extra_runs
    FROM 
        matches m
    JOIN 
        ball_by