```sql
SELECT ws.station_name, ws.state
FROM weather_stations ws
JOIN daily_precipitation dp ON ws.station_id = dp.station_id
WHERE ws.state = 'Washington'
AND YEAR(dp.date) = 2023
AND SUM(CASE WHEN dp.precipitation > 0 AND dp.precipitation != 99.99 THEN 1 ELSE 0 END) > 150
AND YEAR(dp.date) = 2022
AND SUM(CASE WHEN dp.precipitation > 0 AND dp.precipitation != 99.99 THEN 1 ELSE 0 END) < 150;
```

This SQL query selects the names of weather stations in Washington State that recorded more than 150 rainy days in 2023 but fewer rainy days compared to 2022. It uses a `JOIN` to combine the `weather_stations` table with the `daily_precipitation` table based on the station ID. The query filters for stations in Washington State, checks the number of rainy days in 2023 and 2022, and ensures that only stations with valid precipitation data are included. The