SELECT MAX(precipitation) AS max_precipitation, MIN(temperature_min) AS min_temperature, MAX(temperature_max) AS max_temperature 
FROM (
    SELECT YEAR(date) AS year, MONTH(date) AS month, DAY(date) AS day, precipitation, temperature_min, temperature_max 
    FROM weather_data 
    WHERE station_id = 'USW00094846' AND date >= DATE_SUB('2013-01-01', INTERVAL 15 DAY) AND date <= DATE_ADD('2016-12-31', INTERVAL 15 DAY)
      AND precipitation IS NOT NULL AND temperature_min IS NOT NULL AND temperature_max IS NOT NULL