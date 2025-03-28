```sql
SELECT COUNT(T1.station_id)
FROM gsod AS T1
JOIN station AS T2 ON T1.station_id = T2.station_id
WHERE T1.year = 2019
AND T1.temp IS NOT NULL
AND T1.max_temp IS NOT NULL
AND T1.min_temp IS NOT NULL
AND T2.start_date <= '2000-01-01'
AND T2.end_date >= '2019-06-30'
GROUP BY T1.station_id
HAVING SUM(CASE WHEN T1.temp IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) >= 0.9;
```