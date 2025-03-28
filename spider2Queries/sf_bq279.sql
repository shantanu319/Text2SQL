```sql
SELECT 
    YEAR, 
    COUNT(DISTINCT CASE WHEN status = 'active' THEN station_id END) AS active_stations,
    COUNT(DISTINCT CASE WHEN status = 'closed' THEN station_id END) AS closed_stations
FROM 
    bike_share_data
WHERE 
    YEAR IN (2013, 2014)
GROUP BY 
    YEAR;
```