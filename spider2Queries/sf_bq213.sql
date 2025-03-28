```sql
SELECT ipc_code, COUNT(*) AS count
FROM us_b2_patents
WHERE grant_date BETWEEN '2022-06-01' AND '2022-08-31'
GROUP BY ipc_code
ORDER BY count DESC
LIMIT 1;
```