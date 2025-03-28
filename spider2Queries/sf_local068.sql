To achieve the desired result, you can use the following SQL query. This query assumes you have a table named `cities` with columns `city_id`, `year`, `month`, and `inserted_cities`.

```sql
WITH MonthlyTotals AS (
    SELECT 
        YEAR,
        MONTH,
        SUM(inserted_cities) OVER (PARTITION BY YEAR ORDER BY MONTH) AS cumulative_running_total
    FROM 
        cities
    WHERE 
        YEAR IN (2022, 2023)
),
YearOverYearGrowth AS (
    SELECT 
        YEAR,
        MONTH,
        inserted_cities,
        cumulative_running_total,
        LAG(inserted_cities) OVER (ORDER BY YEAR, MONTH) AS prev_year_inserted_cities,
        LAG(cumulative_running_total) OVER (ORDER BY YEAR, MONTH) AS prev_year_cumulative_running_total,
        CASE WHEN prev_year_inserted_cities IS NOT NULL THEN ((inserted_cities - prev_year_inserted_cities) / prev_year_inserted_cities) * 100 ELSE NULL END AS monthly_growth_percentage,
        CASE WHEN prev_year_cumulative_running_total IS NOT NULL THEN ((cumulative_running_total - prev_year_cumulative_running_total