```sql
SELECT 
    l1.category AS category1,
    l2.category AS category2,
    AVG(l1.percentage) AS avg_percentage1,
    AVG(l2.percentage) AS avg_percentage2,
    CORR(l1.percentage, l2.percentage) AS pearson_correlation
FROM 
    iowa_liquor_sales l1
JOIN 
    iowa_liquor_sales l2 ON l1.month = l2.month AND l1.year = l2.year
WHERE 
    l1.category != l2.category
GROUP BY 
    l1.category, l2.category
HAVING 
    COUNT(DISTINCT l1.year) >= 24
ORDER BY 
    pearson_correlation ASC
LIMIT 1;
```