```sql
SELECT MAX(T1.incidents) AS max_incidents
FROM (
    SELECT MONTH(T2.date) AS month, COUNT(*) AS incidents
    FROM motor_vehicle_theft T1
    JOIN incident_dates T2 ON T1.id = T2.motor_vehicle_id
    WHERE YEAR(T2.date) = 2016
    GROUP BY MONTH(T2.date)
) AS monthly_counts;
```