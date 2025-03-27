To convert the given text into an SQL query, you need to identify the relevant tables and columns in your database that store information about weather stations and their locations. Assuming you have two tables: `weather_stations` and `locations`, where `weather_stations` contains columns like `id`, `name`, and `location_id`, and `locations` contains columns like `location_id`, `latitude`, and `longitude`, the SQL query would look something like this:

```sql
SELECT ws.id, ws.name
FROM weather_stations ws
JOIN locations l ON ws.location_id = l.location_id
WHERE 
    l.latitude BETWEEN 41.8319 - 0.5 AND 41.8319 + 0.5
    AND l.longitude BETWEEN 87.6847 - 0.5 AND 87.6847 + 0.5;
```

This query selects the IDs and names of weather stations that are located within a 50 km radius of the center of Chicago. The latitude and longitude values are adjusted by 0.5 degrees to account for the 50 km straight-line distance. Make sure to adjust the table and column names according to your actual database