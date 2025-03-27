```sql
SELECT 
    city,
    class,
    subclass,
    SUM(road_length) AS total_road_length
FROM 
    roads
WHERE 
    quadkey IN ('12020210', '12020211')
    AND (city = 'Amsterdam' OR city = 'Rotterdam')
GROUP BY 
    city, class, subclass;
```

This SQL query calculates the total length of roads for each class and subclass in Amsterdam and Rotterdam, considering only the specified QUADKEY segments. It sums up the `road_length` for each combination of `city`, `class`, and `subclass` that matches the criteria. The results are grouped by these categories to provide a breakdown of the total road lengths for each city and class/subclass pair.