```sql
SELECT 
    t1.pickup_zone,
    t1.dropoff_zone,
    TIMESTAMPDIFF(SECOND, t1.pickup_datetime, t1.dropoff_datetime) AS trip_duration_seconds,
    (t1.trip_distance / TIMESTAMPDIFF(SECOND, t1.pickup_datetime, t1.dropoff_datetime)) * 3600 AS driving_speed_mph,
    (t1.tip_amount / t1.total_fare) * 100 AS tip_rate_percentage
FROM 
    taxi_trips t1
WHERE 
    t1.pickup_datetime BETWEEN '2016-07-01' AND '2016-07-07'
    AND t1.dropoff_datetime > t1.pickup_datetime
    AND t1.passenger_count > 5
    AND t1.trip_distance >= 10
    AND t1.fare_amount >= 0
    AND t1.mta_tax >= 0
    AND t1.tolls_amount >= 0
    AND t1.tip_amount >= 0
ORDER BY 
    t1.total_fare DESC
LIMIT 10;
```