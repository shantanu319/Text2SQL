```sql
-- Step 1: Extract daily new COVID-19 case counts for March to May 2020
SELECT 
    date,
    state,
    new_cases
FROM 
    covid_data
WHERE 
    date BETWEEN '2020-03-01' AND '2020-05-31';

-- Step 2: Identify the top five states by daily increases
WITH DailyTopFive AS (
    SELECT 
        state,
        SUM(new_cases) AS total_new_cases
    FROM 
        (SELECT 
             state,
             new_cases,
             ROW_NUMBER() OVER (PARTITION BY state ORDER BY new_cases DESC) AS rn
         FROM 
             covid_data
         WHERE 
             date BETWEEN '2020-03-01' AND '2020-05-31') AS ranked_states
    WHERE 
        rn <= 5
    GROUP BY 
        state
)
SELECT 
    state,
    total_new_cases
FROM 
    DailyTopFive;

-- Step 3: Compile a ranking based on how often each state appears in these daily top fives
WITH StateFrequency AS (
    SELECT 
        state,
        COUNT(*)