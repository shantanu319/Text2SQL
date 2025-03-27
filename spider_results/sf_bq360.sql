```sql
WITH SpecializationCounts AS (
    SELECT 
        specialization,
        COUNT(DISTINCT npi) AS npi_count
    FROM 
        healthcare_providers
    WHERE 
        practice_location = 'Mountain View, CA'
        AND specialization IN (SELECT specialization FROM healthcare_provider_taxonomy)
    GROUP BY 
        specialization
),
AverageNPICount AS (
    SELECT 
        AVG(npi_count) AS avg_npi_count
    FROM 
        SpecializationCounts
)
SELECT 
    specialization
FROM 
    SpecializationCounts
WHERE 
    npi_count = (
        SELECT 
            MIN(abs(avg_npi_count - npi_count))
        FROM 
            AverageNPICount
    )
ORDER BY 
    npi_count DESC
LIMIT 1;
```