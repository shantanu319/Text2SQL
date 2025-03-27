```sql
SELECT COUNT(*) AS num_patents
FROM us_b2_patents
WHERE grant_date BETWEEN '2008-01-01' AND '2018-12-31'
  AND NOT EXISTS (
    SELECT *
    FROM patent_claims
    WHERE patent_id = us_b2_patents.id
      AND claim_text LIKE '%claim%'
  );
```