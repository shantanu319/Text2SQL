```sql
SELECT 
    T1.city,
    T1.zip_code,
    T1.state,
    COUNT(T2.hurricane_name) AS count_of_hurricanes,
    GROUP_CONCAT(T2.hurricane_name ORDER BY T2.hurricane_name ASC SEPARATOR ', ') AS hurricane_list
FROM 
    cities AS T1
JOIN 
    hurricanes AS T2 ON T1.zip_code = T2.zip_code
WHERE 
    T2.hurricane_name IS NOT NULL AND T2.zip_code BETWEEN T1.zip_code - 50 AND T1.zip_code + 50
GROUP BY 
    T1.city, T1.zip_code, T1.state
ORDER BY 
    count_of_hurricanes DESC;
```