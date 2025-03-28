```sql
SELECT 
    bike_trip_id,
    start_zip_code,
    end_zip_code
FROM 
    bike_trips
WHERE 
    city = 'New York City'
    AND date = '2015-07-15'
    AND start_zip_code IN (
        SELECT 
            zip_code
        FROM 
            weather_data
        WHERE 
            station_id = '94728'
            AND date = '2015-07-15'
        ORDER BY 
            avg_temperature DESC
        LIMIT 1
    )
    AND end_zip_code IN (
        SELECT 
            zip_code
        FROM 
            weather_data
        WHERE 
            station_id = '94728'
            AND date = '2015-07-15'
        ORDER BY 
            avg_temperature DESC
        LIMIT 1
    )
ORDER BY 
    start_zip_code ASC,
    end_zip_code DESC
LIMIT 1;
```