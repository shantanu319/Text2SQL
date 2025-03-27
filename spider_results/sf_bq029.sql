```sql
SELECT 
    YEAR(T2.filed_date) AS filing_year,
    COUNT(*) AS num_patents,
    AVG(CASE WHEN T3.inventor_count > 0 THEN 1 ELSE 0 END) AS avg_inventors_per_patent
FROM 
    patent_publication AS T1
JOIN 
    patent_filing AS T2 ON T1.publication_id = T2.publication_id
LEFT JOIN (
    SELECT 
        publication_id,
        COUNT(DISTINCT inventor_name) AS inventor_count
    FROM 
        patent_inventor_detail
    GROUP BY 
        publication_id
) AS T3 ON T1.publication_id = T3.publication_id
WHERE 
    T2.filed_date >= '1960-01-01' AND T2.filed_date <= '2020-12-31'
GROUP BY 
    YEAR(T2.filed_date)
HAVING 
    YEAR(T2.filed_date) % 5 = 0;
```