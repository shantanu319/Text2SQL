```sql
SELECT country_name, (recovered_cases / confirmed_cases) * 100 AS recovery_rate
FROM covid_data
WHERE confirmed_cases > 50000
ORDER BY recovery_rate DESC
LIMIT 3;
```