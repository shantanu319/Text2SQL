Certainly! To calculate the average number of new unicorn companies per year in the top industry from 2019 to 2021, you would typically need a database table that contains information about unicorn companies and their industries. Let's assume you have a table named `unicorn_companies` with the following relevant columns:

- `company_name`: The name of the unicorn company.
- `industry`: The industry of the unicorn company.
- `year_founded`: The year the unicorn company was founded.

Here is an example SQL query to achieve your goal:

```sql
SELECT AVG(CASE WHEN YEAR = '2019' THEN 1 ELSE 0 END) AS avg_2019,
       AVG(CASE WHEN YEAR = '2020' THEN 1 ELSE 0 END) AS avg_2020,
       AVG(CASE WHEN YEAR = '2021' THEN 1 ELSE 0 END) AS avg_2021
FROM (
    SELECT company_name, industry, YEAR(year_founded) AS YEAR
    FROM unicorn_companies
    WHERE industry IN (SELECT TOP 1 industry FROM unicorn_companies GROUP BY industry ORDER BY COUNT(*) DESC)
) AS filtered_companies;