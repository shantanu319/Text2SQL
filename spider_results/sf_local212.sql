```sql
SELECT driver_id, AVG(daily_deliveries) AS avg_daily_deliveries
FROM deliveries
GROUP BY driver_id
ORDER BY avg_daily_deliveries DESC
LIMIT 5;
```