```sql
SELECT 
    objectId,
    title,
    DATE_FORMAT(metadataDate, '%Y-%m-%d') AS formattedMetadataDate
FROM 
    objects
WHERE 
    department = 'The Libraries'
    AND cropConfidence > 0.5
    AND title LIKE '%book%';
```