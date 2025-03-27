```sql
SELECT 
    COUNT(CASE WHEN T2.app_exception = 'true' THEN 1 ELSE NULL END) * 100 / COUNT(T1.user_id) AS crash_percentage
FROM 
    user_openings T1
JOIN 
    user_uninstallations T2 ON T1.user_id = T2.user_id
WHERE 
    T1.opening_date BETWEEN '2018-09-01' AND '2018-09-30'
    AND DATEDIFF(T2.uninstallation_date, T1.opening_date) <= 7;
```

This SQL query calculates the percentage of users who experienced an app crash after first opening the app in September 2018 and then uninstalling within seven days. It joins two tables: `user_openings` and `user_uninstallations`, filters the data based on the specified conditions, and computes the required percentage.