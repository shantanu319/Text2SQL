```sql
SELECT 
    t.borough,
    COUNT(CASE WHEN t.tip_rate = 0 THEN 1 ELSE NULL END) AS no_tip_count,
    COUNT(*) AS total_trip_count,
    CAST(COUNT(CASE WHEN t.tip_rate = 0 THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(*) AS no_tip_percentage
FROM 
    (
        SELECT 
            t.borough,
            CASE 
                WHEN t.tip_amount > 0 AND t.total_amount > 0 THEN (t.tip_amount * 100) / t.total_amount
                ELSE 0
            END AS tip_rate
        FROM 
            yellow_taxi_trips t
        WHERE 
            t.pickup_datetime BETWEEN '2016-01-01' AND '2016-01-07'
            AND t.dropoff_datetime > t.pickup_datetime
            AND t.passenger_count > 0
            AND t.trip_distance >= 0
            AND t.tip_amount >= 0
            AND t.tolls_amount >= 0
            AND t.MTA_tax >= 0
            AND t.fare_amount >= 0
            AND t.total_amount >=