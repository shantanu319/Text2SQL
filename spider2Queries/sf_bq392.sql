```sql
SELECT DATE_FORMAT(date_column, '%Y-%m-%d') AS date,
       AVG(temperature_column) AS avg_temp
FROM weather_data
WHERE station_number = 723758 AND MONTH(date_column) = 10 AND YEAR(date_column) = 2009
GROUP BY DATE_FORMAT(date_column, '%Y-%m-%d')
ORDER BY avg_temp DESC
LIMIT 3;
```

Replace `date_column`, `temperature_column`, and `weather_data` with the actual column names and table name used in your database. Also, ensure that the date format matches the one used in your data. If your date format is different, adjust the `DATE_FORMAT` function accordingly. Additionally, if you need to handle time zones or daylight saving times, consider using functions like `STRFTIME` or `TIMEZONE` depending on your database system.