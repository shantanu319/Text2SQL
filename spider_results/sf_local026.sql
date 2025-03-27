To achieve the desired result using SQL, you need to follow several steps. The task involves identifying the bowler with the highest number of runs conceded in a single over for each match where the maximum runs were conceded. Here's a step-by-step breakdown of how you can construct the SQL query:

1. **Identify Matches with Maximum Runs**: First, find the matches where the maximum runs were conceded.

2. **Find Bowlers with Highest Runs in Single Over**: For each match identified in step 1, find the bowler who conceded the highest number of runs in a single over.

3. **Combine Results**: Combine the results from step 2 to get the top 3 bowlers and the matches where they conceded the maximum runs.

Here's the SQL query to accomplish this:

```sql
WITH MaxRunsPerMatch AS (
    SELECT 
        MatchID,
        MAX(RunsConceded) AS MaxRuns
    FROM 
        Overs
    GROUP BY 
        MatchID
),
BowlersWithMaxRuns AS (
    SELECT 
        MatchID,
        BowlerID,
        RunsConcededInSingleOver
    FROM 
        Overs
    WHERE 
        (MatchID, RunsConcededInSingleOver) IN (