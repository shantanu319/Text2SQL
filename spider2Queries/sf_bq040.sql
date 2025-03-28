```sql
SELECT 
    pickup_borough,
    CASE 
        WHEN tip_percentage = 0 THEN '0%'
        WHEN tip_percentage BETWEEN 1 AND 5 THEN 'up to 5%'
        WHEN tip_percentage BETWEEN 6 AND 10 THEN '5% to 10%'
        WHEN tip_percentage BETWEEN 11 AND 15 THEN '10% to 15%'
        WHEN tip_percentage BETWEEN 16 AND 20 THEN '15% to 20%'
        WHEN tip_percentage BETWEEN 21 AND 25 THEN '20% to 25%'
        ELSE '> 25%'
    END AS tip_category,
    COUNT(*) AS ride_count
FROM (
    SELECT 
        t1.pickup_borough,
        t1.tip_percentage
    FROM (
        SELECT 
            t2.pickup_borough,
            SUM(CASE WHEN t3.tip_amount > 0 THEN 1 ELSE 0 END) / COUNT(t3.id) * 100 AS tip_percentage
        FROM (
            SELECT 
                t4.id,
                t4.pickup_borough,
                t4.dropoff_time,
                t4.passenger_count,
                t4.trip_distance,
                t