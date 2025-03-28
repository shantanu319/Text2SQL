```sql
SELECT T1.*
FROM patents AS T1
JOIN backward_citations AS T2 ON T1.id = T2.patent_id
WHERE T1.kind_code = 'B2'
  AND T1.grant_date BETWEEN '2015-01-01' AND '2018-12-31'
ORDER BY 1 - SUM(SQRT(CASE WHEN COUNT(DISTINCT SUBSTR(T2.ipc_code, 1, 4)) > 0 THEN 1 ELSE 0 END)) / SQRT(COUNT(DISTINCT SUBSTR(T2.ipc_code, 1, 4))) DESC
LIMIT 1;
```