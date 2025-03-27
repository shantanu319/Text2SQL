```sql
SELECT 
    AVG(anxiety_level) AS avg_anxiety_2019,
    AVG(depression_level) AS avg_depression_2019,
    AVG(anxiety_level) - AVG(anxiety_level) OVER (PARTITION BY YEAR(date_column) ORDER BY YEAR(date_column)) * 100 / AVG(anxiety_level) AS pct_increase_anxiety,
    AVG(depression_level) - AVG(depression_level) OVER (PARTITION BY YEAR(date_column) ORDER BY YEAR(date_column)) * 100 / AVG(depression_level) AS pct_increase_depression
FROM 
    weekly_country_data
WHERE 
    date_column BETWEEN '2019-01-01' AND '2020-01-01'
UNION ALL
SELECT 
    AVG(anxiety_level) AS avg_anxiety_2020,
    AVG(depression_level) AS avg_depression_2020,
    AVG(anxiety_level) - AVG(anxiety_level) OVER (PARTITION BY YEAR(date_column) ORDER BY YEAR(date_column)) * 100 / AVG(anxiety_level) AS pct_increase_anxiety,
    AVG(depression_level) - AVG(depression_level