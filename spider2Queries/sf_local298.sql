To achieve the desired result using SQL, you can use a combination of window functions and conditional logic. Here's how you can write the query:

```sql
WITH MonthlyBalances AS (
    SELECT 
        DATE_TRUNC('month', created_at) AS month,
        user_id,
        SUM(balance) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_balance
    FROM 
        transactions
),
PreviousMonthBalances AS (
    SELECT 
        month,
        user_id,
        CASE 
            WHEN running_balance < 0 THEN 0
            ELSE running_balance
        END AS previous_month_balance
    FROM 
        MonthlyBalances
)
SELECT 
    month,
    SUM(previous_month_balance) AS total_previous_month_balance
FROM 
    PreviousMonthBalances
GROUP BY 
    month
ORDER BY 
    month;
```

### Explanation:

1. **MonthlyBalances CTE**: This common table expression calculates the running balance for each user per month. The `SUM` function is used with a window frame to accumulate the balance over time.

2. **PreviousMonthBalances CTE**: This CTE filters out any negative balances and replaces them with zero. It