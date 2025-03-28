To achieve the desired result, you can use the following SQL query. This query assumes you have a table named `salary_dataset` with at least the following columns: `company_name`, `location`, `salary`.

```sql
WITH CleanedSalaries AS (
    SELECT 
        company_name,
        location,
        REPLACE(REPLACE(salary, ',', ''), '$', '') AS cleaned_salary
    FROM 
        salary_dataset
),
NumericSalaries AS (
    SELECT 
        company_name,
        location,
        CAST(CAST(cleaned_salary AS DECIMAL(10, 2)) AS INT) AS numeric_salary
    FROM 
        CleanedSalaries
),
CompanyAverages AS (
    SELECT 
        location,
        company_name,
        AVG(numeric_salary) AS avg_salary_in_state
    FROM 
        NumericSalaries
    GROUP BY 
        location, company_name
),
NationalAverage AS (
    SELECT 
        AVG(numeric_salary) AS avg_salary_in_country
    FROM 
        NumericSalaries
)
SELECT 
    ca.location,
    ca.company_name,
    ca.avg_salary_in_state,
    na.avg_salary_in_country
FROM 
    CompanyAverages ca
JOIN 
    NationalAverage na ON 1 = 1