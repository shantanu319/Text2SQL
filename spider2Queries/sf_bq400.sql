```sql
SELECT 
    T1.trip_headsign,
    MIN(T2.departure_time) AS earliest_departure,
    MAX(T3.arrival_time) AS latest_arrival
FROM 
    stops AS T1
JOIN 
    trips AS T2 ON T1.stop_id = T2.start_stop_id
JOIN 
    trips AS T3 ON T1.stop_id = T3.end_stop_id
WHERE 
    T1.name IN ('Clay St & Drumm St', 'Sacramento St & Davis St')
    AND T2.sequence < T3.sequence
GROUP BY 
    T1.trip_headsign;
```

This SQL query retrieves the earliest departure time from 'Clay St & Drumm St' and the latest arrival time at 'Sacramento St & Davis St' for each trip, formatted as HH:MM:SS. It also includes the trip headsign for each route. The query assumes that the `stops` table contains information about the stops, the `trips` table contains information about the trips, including their start and end stops and sequences, and that the `departure_time` and `arrival_time` columns are present in the `trips` table. Adjust the column names and table names as necessary to fit your