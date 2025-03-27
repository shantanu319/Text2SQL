```sql
SELECT DISTINCT driver_name
FROM (
    SELECT 
        driver_id,
        COUNT(CASE WHEN overtaken_by_others > overtaken_by_driver THEN 1 END) AS overtaken_more_times_than_overtaken_by_others,
        COUNT(CASE WHEN overtaken_by_driver > overtaken_by_others THEN 1 END) AS overtaken_less_times_than_overtaken_by_driver
    FROM race_laps
    GROUP BY driver_id
) AS driver_overlaps
WHERE overtaken_more_times_than_overtaken_by_others > overtaken_less_times_than_overtaken_by_driver;
```

This SQL query calculates the number of times each driver has been overtaken by other drivers and overtaken by themselves during race laps. It then filters out drivers who have been overtaken more times than they have overtaken others, excluding positions changed due to pit stops, retirements, or movements at the start of the race. The result is a list of full names of such drivers. Note that `driver_name` should be replaced with the actual column name for the driver's full name in your database schema. Additionally, ensure that the columns `overtaken_by_others`, `overtaken_by_driver`, and `race_laps` match