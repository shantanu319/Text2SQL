```sql
SELECT 
    n.neighborhood_name,
    COUNT(bs.station_id) AS total_bike_share_stations,
    COUNT(ci.incident_id) AS total_crime_incidents
FROM 
    neighborhoods n
JOIN 
    bike_share_stations bs ON n.neighborhood_id = bs.neighborhood_id
JOIN 
    crime_incidents ci ON n.neighborhood_id = ci.neighborhood_id
GROUP BY 
    n.neighborhood_name
HAVING 
    COUNT(bs.station_id) > 0 AND COUNT(ci.incident_id) > 0;
```