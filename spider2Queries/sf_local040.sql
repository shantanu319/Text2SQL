```sql
SELECT 
    Borough,
    AVG(MeanIncome) AS AverageMeanIncome
FROM (
    SELECT 
        T1.Borough,
        T2.MedianIncome,
        T2.MeanIncome
    FROM 
        TreesData AS T1
    LEFT JOIN 
        IncomeData AS T2 ON T1.ZIPCode = T2.ZIPCode
    WHERE 
        T2.MedianIncome > 0 AND T2.MeanIncome > 0 AND T1.Borough IS NOT NULL
) AS SubQuery
GROUP BY 
    Borough
ORDER BY 
    COUNT(*) DESC
LIMIT 3;
```