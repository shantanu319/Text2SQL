```sql
SELECT 
    TRUNC(TripDuration / 60) AS TripDurationGroup,
    MIN(TripDuration) AS MinTripDuration,
    MAX(TripDuration) AS MaxTripDuration,
    COUNT(*) AS TotalTrips,
    AVG(FareAmount) AS AverageFare
FROM 
    Trips
WHERE 
    TripDuration BETWEEN 0 AND 60
GROUP BY 
    TRUNC(TripDuration / 60)
ORDER BY 
    TripDurationGroup;
```

This SQL query calculates the minimum and maximum trip durations in minutes (rounded to the nearest whole number), total number of trips, and average fare for each of six equal quantile groups based on trip duration, considering only trips between 0 and 60 minutes. The `TRUNC` function is used to round down the trip duration to the nearest whole number before grouping. The results are ordered by the trip duration group.