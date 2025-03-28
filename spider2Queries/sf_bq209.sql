```sql
SELECT COUNT(T1.patent_id) AS num_utility_patents
FROM patent AS T1
JOIN citation AS T2 ON T1.patent_id = T2.cited_patent_id
WHERE T1.application_date BETWEEN '2010-01-01' AND '2020-01-01'
  AND T2.citation_date BETWEEN '2010-01-01' AND '2020-01-01'
GROUP BY T1.patent_id
HAVING COUNT(DISTINCT T2.citation_date) = 1;
```