```sql
SELECT MAX(distance) AS longest_route_distance
FROM routes
WHERE departure_city = 'Abakan' OR destination_city = 'Abakan';
```