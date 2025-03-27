```sql
SELECT 
    p.period,
    l.label,
    COUNT(*) AS associated_count
FROM 
    artworks a
JOIN 
    periods p ON a.period_id = p.id
JOIN 
    labels l ON a.label_id = l.id
GROUP BY 
    p.period, l.label
HAVING 
    COUNT(*) >= 500
ORDER BY 
    p.period, associated_count DESC
LIMIT 3;
```