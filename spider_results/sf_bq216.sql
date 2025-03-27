```sql
SELECT DISTINCT T2.publication_number
FROM patent AS T1
JOIN patent_similarity AS T3 ON T1.id = T3.patent_id_1
JOIN patent AS T2 ON T3.patent_id_2 = T2.id
WHERE T1.application_number = 'US-9741766-B2'
  AND T2.year = T1.year
ORDER BY T3.similarity_score DESC
LIMIT 5;
```