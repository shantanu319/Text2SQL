```sql
SELECT 
    T1.publication_number,
    T1.ipc_code_4
FROM 
    us_patent_b2 AS T1
WHERE 
    T1.grant_date BETWEEN '2022-06-01' AND '2022-09-30'
GROUP BY 
    T1.publication_number, 
    T1.ipc_code_4
HAVING 
    COUNT(*) >= 10;
```