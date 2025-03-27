```sql
SELECT 
    r.region_name,
    rd.trip_id,
    rd.duration,
    rd.start_time,
    rs.station_name AS start_station,
    rd.gender
FROM 
    rides rd
JOIN 
    regions r ON rd.region_id = r.region_id
JOIN 
    stations rs ON rd.start_station_id = rs.station_id
WHERE 
    rd.year BETWEEN 2014 AND 2017
ORDER BY 
    r.region_name,
    rd.year DESC;
```

This query joins the `rides`, `regions`, and `stations` tables to retrieve the required information. It filters the rides to include only those from 2014 to 2017 and orders the results by region and year in descending order to get the latest ride data for each region. The `gender` column is included as requested.