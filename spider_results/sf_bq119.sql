SELECT h.name, h.travel_coordinates, SUM(h.cumulative_travel_distance), MAX(h.max_sustained_wind_speed)
FROM hurricanes h
WHERE h.year = 2020 AND h.region = 'North Atlantic'
GROUP BY h.name