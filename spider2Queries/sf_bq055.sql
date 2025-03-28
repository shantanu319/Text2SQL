```sql
SELECT race_name, 
       (google_percentage - bls_average_percentage) AS percentage_difference,
       google_percentage,
       bls_average_percentage
FROM (
    SELECT race_name,
           AVG(hiring_data) AS google_percentage
    FROM (
        SELECT race_name,
               CASE WHEN job_title LIKE '%Google%' THEN 1 ELSE 0 END AS hiring_data
        FROM your_table_name
        WHERE year = 2021 AND job_title NOT IN ('Dar', 'Non Intersectional Hiring')
    ) AS subquery
    GROUP BY race_name
) AS google_races
JOIN (
    SELECT race_name,
           AVG(bls_percentage) AS bls_average_percentage
    FROM (
        SELECT race_name,
               CASE WHEN industry_group IN ('Internet publishing and broadcasting and web search portals',
                                           'Software publishers',
                                           'Data processing, hosting, and related services',
                                           'Computer systems design and related services') THEN 1 ELSE 0 END AS bls_percentage
        FROM your_table_name
        WHERE year = 2021 AND industry_group IN ('Internet publishing and broadcasting and web search portals',
                                                  'Software publishers',
                                                  'Data processing, hosting, and related services',
                                                  'Computer systems design and related