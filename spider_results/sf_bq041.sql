To compute the monthly statistics for new Stack Overflow users created in 2021, you can use the following SQL query:

```sql
WITH MonthlyStats AS (
    SELECT 
        DATE_TRUNC('month', signup_date) AS month,
        COUNT(*) AS total_new_users,
        SUM(CASE WHEN asked_question_within_30_days THEN 1 ELSE 0 END) AS asked_question_count,
        SUM(CASE WHEN answered_question_within_30_days THEN 1 ELSE 0 END) AS answered_question_count
    FROM 
        stackoverflow_users
    WHERE 
        YEAR(signup_date) = 2021
    GROUP BY 
        DATE_TRUNC('month', signup_date)
),
PercentageStats AS (
    SELECT 
        month,
        (asked_question_count / total_new_users) * 100 AS asked_percentage,
        (answered_question_count / asked_question_count) * 100 AS answered_percentage
    FROM 
        MonthlyStats
)
SELECT 
    month,
    total_new_users,
    asked_percentage,
    answered_percentage
FROM 
    PercentageStats;
```

### Explanation:
1. **MonthlyStats CTE**: This Common Table Expression calculates the total number of new users per month, the count of