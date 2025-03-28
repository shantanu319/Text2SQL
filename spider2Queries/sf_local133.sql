To achieve the desired result using SQL, you can follow these steps:

1. **Create Tables**: First, ensure you have the necessary tables to store the data. For example:
   - `Users` table with columns: `User_ID`, `First_Style`, `Second_Style`, `Third_Style`.
   - `MusicalStyles` table with columns: `Style_ID`, `Style_Name`.

2. **Calculate Weighted Scores**: Use a common table expression (CTE) or subquery to calculate the weighted scores for each user.

3. **Aggregate Scores**: Group the results by `Style_ID` to calculate the total weighted score for each style.

4. **Compute Average Score**: Calculate the average total weighted score across all styles.

5. **Compute Absolute Differences**: Finally, compute the absolute difference between each style's total weighted score and the average score.

Here is the SQL query to perform these steps:

```sql
WITH UserScores AS (
    SELECT 
        U.User_ID,
        U.First_Style,
        U.Second_Style,
        U.Third_Style,
        CASE 
            WHEN U.First_Style IS NOT NULL THEN 3
            ELSE 0
        END + 
        CASE 
            WHEN U.Second_Style IS NOT NULL THEN 2