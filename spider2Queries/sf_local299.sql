To achieve the desired result using SQL, we need to break down the task into several steps:

1. Calculate the average balance for each user over the past 30 days, computed daily.
2. For each month, find the highest of these daily averages for each user.
3. Sum these maximum values across all users for each month.
4. Exclude the first month from the output.

Here's how you can write the SQL query to accomplish this:

```sql
WITH DailyBalances AS (
    SELECT 
        user_id,
        DATE_TRUNC('day', transaction_date) AS transaction_day,
        AVG(balance) AS daily_average_balance
    FROM 
        transactions
    WHERE 
        transaction_date >= NOW() - INTERVAL '30 days'
    GROUP BY 
        user_id, DATE_TRUNC('day', transaction_date)
),
MonthlyAverages AS (
    SELECT 
        user_id,
        EXTRACT(YEAR_MONTH FROM transaction_day) AS year_month,
        MAX(daily_average_balance) AS max_daily_average
    FROM 
        DailyBalances
    GROUP BY 
        user_id, EXTRACT(YEAR_MONTH FROM transaction_day)
),
FinalResult AS (
    SELECT 
        year_month,
        SUM(max_daily_average) AS total_max_daily_average