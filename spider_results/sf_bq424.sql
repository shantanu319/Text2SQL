```sql
SELECT country_name, SUM(long_term_external_debt) AS total_debt
FROM countries
WHERE region IS NOT NULL
GROUP BY country_name
ORDER BY total_debt DESC
LIMIT 10;
```