```sql
SELECT COUNT(DISTINCT family_id) AS number_of_families_with_over_one_application
FROM patents
WHERE grant_date BETWEEN '2010-01-01' AND '2023-12-31'
GROUP BY family_id
HAVING COUNT(application_id) > 1;
```