```sql
SELECT AVG(prenatal_weeks) AS average_prenatal_weeks
FROM county_data
WHERE state = 'Wisconsin'
  AND YEAR(employment_data.year) = 2017
  AND employment_data.commute_minutes BETWEEN 45 AND 59
  AND employment_data.percent_employed > 5;
```