```sql
WITH TopFamilies AS (
    SELECT family_id, COUNT(*) AS num_publications
    FROM publications
    WHERE family_id != '-1'
    GROUP BY family_id
    ORDER BY num_publications DESC
    LIMIT 6
),
FamilyPublications AS (
    SELECT fp.family_id, ap.abstract
    FROM TopFamilies tf
    JOIN publications p ON tf.family_id = p.family_id
    LEFT JOIN abs_and_emb ae ON p.publication_number = ae.publication_number
)
SELECT family_id, abstract
FROM FamilyPublications;
```