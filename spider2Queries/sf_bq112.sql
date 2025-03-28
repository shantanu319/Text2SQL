```sql
SELECT 
    AVG(wage_growth_rate) AS avg_wage_growth_rate,
    AVG(cpi_growth_rate) AS avg_cpi_growth_rate
FROM (
    SELECT 
        YEAR(date) AS year,
        SUM(CASE WHEN industry = 'All Industries' THEN wage ELSE 0 END) / COUNT(*) AS avg_annual_wages,
        SUM(CASE WHEN industry = 'All Industries' THEN cpi ELSE 0 END) / COUNT(*) AS avg_annual_cpi,
        (SUM(CASE WHEN industry = 'All Industries' THEN wage ELSE 0 END) / COUNT(*)) - (SUM(CASE WHEN industry = 'All Industries' THEN cpi ELSE 0 END) / COUNT(*)) AS wage_growth_rate,
        (SUM(CASE WHEN industry = 'All Industries' THEN cpi ELSE 0 END) / COUNT(*)) - (SUM(CASE WHEN industry = 'All Industries' THEN cpi ELSE 0 END) / COUNT(*)) AS cpi_growth_rate
    FROM 
        employment_data
    WHERE 
        county = 'Allegheny'
        AND city = 'Pittsburgh'
        AND YEAR(date) BETWEEN 1998 AND 2017
    GROUP BY