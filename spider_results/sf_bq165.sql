```sql
SELECT 
    chr,
    start_bp,
    end_bp,
    COUNT(*) AS total_samples,
    ROUND((COUNT(*) * 100 / (SELECT COUNT(*) FROM breast_cancer WHERE morphology = '3111' AND topology = '0401')), 2) AS frequency_percentage,
    SUM(CASE WHEN copy_number > 1 THEN 1 ELSE 0 END) AS amplifications,
    SUM(CASE WHEN copy_number = 2 THEN 1 ELSE 0 END) AS gains,
    SUM(CASE WHEN copy_number = 1 THEN 1 ELSE 0 END) AS losses,
    SUM(CASE WHEN copy_number = 0 THEN 1 ELSE 0 END) AS homozygous_deletions
FROM 
    cytogenetic_coordinates
WHERE 
    sample_id IN (
        SELECT 
            sample_id 
        FROM 
            breast_cancer 
        WHERE 
            morphology = '3111' AND topology = '0401'
    )
GROUP BY 
    chr, start_bp, end_bp
ORDER BY 
    chr, start_bp;
```