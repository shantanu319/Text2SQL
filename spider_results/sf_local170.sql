```sql
SELECT T3.state_abbr
FROM (
    SELECT T1.leg_id, T1.term_start_date, T1.term_end_date, T1.gender, 
           (T1.term_end_date - T1.term_start_date) AS term_length,
           CASE WHEN T1.term_length <= 10 THEN 1 ELSE 0 END AS within_first_10_years
    FROM legislators AS T1
    WHERE T1.term_start_date >= '2000-01-01' AND T1.term_end_date <= '2010-12-31'
) AS T2
JOIN states AS T3 ON T2.state_id = T3.state_id
WHERE T2.within_first_10_years = 1
GROUP BY T3.state_abbr
HAVING AVG(CASE WHEN DATEDIFF(T2.term_end_date, T2.term_start_date) BETWEEN 0 AND 10 THEN 1 ELSE 0 END) > 0;
```

This SQL query identifies the state abbreviations where the retention rate of the initial cohort is greater than zero at each of the intervals 0, 2, 4, 6, 8, and 10