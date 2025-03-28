```sql
SELECT AVG(salary) AS overall_average_salary
FROM (
    SELECT DISTINCT skill
    FROM job_postings
    WHERE position = 'Data Analyst'
      AND annual_average_salary IS NOT NULL
      AND location = 'remote'
    ORDER BY COUNT(*) DESC
    LIMIT 3
) AS top_skills
JOIN job_postings ON top_skills.skill = job_postings.skill
WHERE job_postings.position = 'Data Analyst'
  AND job_postings.annual_average_salary IS NOT NULL
  AND job_postings.location = 'remote';
```