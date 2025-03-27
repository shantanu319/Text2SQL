```sql
SELECT 
    city,
    MAX(1990_pm2_5) - MIN(2020_pm2_5) AS pm2_5_difference
FROM (
    SELECT 
        city,
        1990_pm2_5,
        2020_pm2_5
    FROM (
        SELECT 
            city,
            measurement_value AS 1990_pm2_5,
            NULL AS 2020_pm2_5
        FROM 
            your_dataset_1990
        WHERE 
            units_of_measure = 'Micrograms/cubic meter (LC)'
            AND parameter_name = 'Acceptable PM2.5 AQI & Speciation Mass'
            AND YEAR(timestamp_column) = 1990
    ) AS dataset_1990
    FULL OUTER JOIN (
        SELECT 
            city,
            measurement_value AS 2020_pm2_5,
            NULL AS 1990_pm2_5
        FROM 
            your_dataset_2020
        WHERE 
            pollutant = 'pm25'
            AND YEAR(timestamp_column) = 2020
    ) AS dataset_