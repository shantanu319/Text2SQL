```sql
SELECT problem_id
FROM (
    SELECT problem_id,
           COUNT(*) AS total_occurrences
    FROM solution_table
    WHERE step IN (1, 2, 3)
      AND model != 'Stack'
    GROUP BY problem_id
) AS subquery
WHERE total_occurrences > (
    SELECT MAX(total_occurrences)
    FROM (
        SELECT problem_id,
               COUNT(*) AS total_occurrences
        FROM solution_table
        WHERE step IN (1, 2, 3)
          AND model = 'Stack'
        GROUP BY problem_id
    ) AS subquery
);
```