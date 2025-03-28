```sql
SELECT ROUND(AVG(speed), 1) AS avg_speed
FROM trips
WHERE city = 'Berkeley' AND distance > 1000;
```