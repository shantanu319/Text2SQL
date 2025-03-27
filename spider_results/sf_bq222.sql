```sql
SELECT 
    T3.full_title,
    T3.cpc_group,
    T2.year,
    T2.exponential_moving_average
FROM 
    germany_patent_data AS T1
JOIN 
    germany_exponential_moving_average AS T2 ON T1.id = T2.patent_id
JOIN 
    cpc_groups AS T3 ON T1.cpc_group_id = T3.id
WHERE 
    T1.grant_date LIKE '2016-12-%'
GROUP BY 
    T3.full_title, T3.cpc_group, T2.year, T2.exponential_moving_average
ORDER BY 
    T2.exponential_moving_average DESC;
```