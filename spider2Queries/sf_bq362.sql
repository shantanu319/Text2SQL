SELECT company, MAX(trip_increase) as max_trip_increase
FROM (
    SELECT company, 
           SUM(CASE WHEN month = 'January' THEN trip_numbers ELSE 0 END) - 
           SUM(CASE WHEN month = 'December' THEN trip_numbers ELSE 0 END) as trip_increase
    FROM trips
    WHERE year = 2018
    GROUP BY company
) AS monthly_changes
WHERE trip_increase > 0
ORDER BY max_trip_increase DESC
LIMIT 3;