```sql
SELECT title, artist_name, medium, original_image_url 
FROM Photographs 
WHERE name LIKE '%Photograph%' 
AND department = 'Photographs' 
AND artist_name != 'Unknown Artist' 
AND object_end_date <= 1839;
```