```sql
SELECT 
    DATE_FORMAT(date, '%m-%d') AS date,
    MAX(covid_confirmed_case_growth_rate) AS max_growth_rate
FROM 
    us_covid_data
WHERE 
    date BETWEEN '2020-03-01' AND '2020-04-30'
GROUP BY 
    DATE_FORMAT(date, '%m-%d')
ORDER BY 
    max_growth_rate DESC
LIMIT 1;
```