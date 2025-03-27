```sql
SELECT 
    MIN(start_position) AS min_start_position,
    MAX(start_position) AS max_start_position,
    SUM(CASE WHEN reference_base = 'AT' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS proportion_at,
    SUM(CASE WHEN reference_base = 'TA' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS proportion_ta
FROM 
    phase_3_human_genetic_variations
WHERE 
    collection_date = '2015-02-20';
```

This SQL query calculates the minimum and maximum start positions of the genetic variations collected in phase 3 on February 20, 2015. It also computes the proportions of the reference bases 'AT' and 'TA' among all the genetic variations. The `SUM` function is used with a `CASE` statement to count occurrences of each reference base, and the result is divided by the total number of variations to get the proportions. The `COUNT(*)` function ensures that the division is valid even if there are no variations of a particular type. Finally, the results are filtered using the `WHERE` clause to match the specified collection date.