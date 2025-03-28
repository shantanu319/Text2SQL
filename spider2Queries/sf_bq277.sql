```sql
SELECT T1.port_name
FROM ports AS T1
JOIN regions AS T2 ON T1.region_id = T2.region_id
WHERE T2.region_number = 6585 AND T1.state_boundary = TRUE
AND T1.geographic_area IN (
    SELECT T3.geographic_area
    FROM named_tropical_storms AS T3
    WHERE T3.wind_speed >= 35 AND T3.basin = 'North Atlantic'
    EXCEPT
    SELECT T4.name
    FROM not_named_tropical_storms AS T4
)
GROUP BY T1.port_name
ORDER BY COUNT(*) DESC
LIMIT 1;
```