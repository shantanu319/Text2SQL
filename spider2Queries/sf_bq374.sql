```sql
SELECT 
    (SUM(CASE WHEN T2.purchase_date IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(T1.user_id)) AS percentage_of_new_users
FROM 
    new_users AS T1
LEFT JOIN 
    user_visits AS T2 ON T1.user_id = T2.user_id AND T2.visit_date BETWEEN '2016-08-01' AND '2017-04-30'
WHERE 
    T1.registration_date BETWEEN '2016-08-01' AND '2017-04-30'
    AND T2.initial_visit_duration > 5
    AND T2.purchase_date IS NOT NULL;
```

This SQL query calculates the percentage of new users who met the specified criteria by joining the `new_users` table with the `user_visits` table based on the user ID and date range. It then filters the results to include only those users who registered within the specified period, had an initial visit duration greater than 5 minutes, and made a purchase on a subsequent visit. Finally, it computes the percentage of such users relative to the total number of new users in