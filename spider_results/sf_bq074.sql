```sql
SELECT COUNT(*) AS NumberOfCounties
FROM (
    SELECT 
        county_name,
        SUM(CASE WHEN YEAR = 2015 THEN unemployment_increase ELSE 0 END) AS unemployment_increase_2015,
        SUM(CASE WHEN YEAR = 2016 THEN unemployment_increase ELSE 0 END) AS unemployment_increase_2016,
        SUM(CASE WHEN YEAR = 2017 THEN unemployment_increase ELSE 0 END) AS unemployment_increase_2017,
        SUM(CASE WHEN YEAR = 2018 THEN unemployment_increase ELSE 0 END) AS unemployment_increase_2018,
        SUM(CASE WHEN YEAR = 2015 THEN dual_eligible_decrease ELSE 0 END) AS dual_eligible_decrease_2015,
        SUM(CASE WHEN YEAR = 2016 THEN dual_eligible_decrease ELSE 0 END) AS dual_eligible_decrease_2016,
        SUM(CASE WHEN YEAR = 2017 THEN dual_eligible_decrease ELSE 0 END) AS dual_eligible_decrease_2017,
        SUM(CASE WHEN YEAR = 2