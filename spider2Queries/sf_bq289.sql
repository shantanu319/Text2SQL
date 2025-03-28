```sql
SELECT 
    amenity,
    ST_Distance(
        ST_GeomFromText('POINT(longitude latitude)', 4326),
        ST_GeomFromText('POINT(other_longitude other_latitude)', 4326)
    ) AS distance
FROM 
    (
        SELECT 
            amenity,
            longitude,
            latitude
        FROM 
            pennsylvania_table
        WHERE 
            amenity IN ('library', 'place_of_worship', 'community_center')
            AND ST_Within(ST_GeomFromText('POINT(longitude latitude)', 4326), planet_features_points)
    ) AS amenities
ORDER BY 
    distance ASC
LIMIT 1;
```

This query calculates the shortest distance between any two amenities in Philadelphia by comparing their coordinates. It uses the `ST_Distance` function to compute the distance between two points on the Earth's surface. The results are ordered by distance and limited to the shortest one found. Make sure to replace `'longitude latitude'`, `'other_longitude other_latitude'`, and the column names with actual data from your tables. Also, ensure that the `planet_features_points` table contains the necessary geometry information for the comparison.