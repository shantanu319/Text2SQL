```sql
SELECT 
    COUNT(CASE WHEN return_date IS NOT NULL THEN 1 ELSE NULL END) / COUNT(*) AS return_rate
FROM 
    user_cohort
WHERE 
    first_purchase_date BETWEEN '2020-01-01' AND '2020-01-31'
    AND return_date BETWEEN '2020-02-01' AND '2020-12-31';
```

This query calculates the proportion of users who returned in the subsequent months of 2020 for the user cohort that had their first purchase in January 2020. It uses conditional counting to determine if a user returned and then divides by the total number of users in the cohort to get the return rate. The `return_date` column is assumed to be present in the `user_cohort` table, indicating when each user returned. Adjust the column names as necessary based on your actual database schema.