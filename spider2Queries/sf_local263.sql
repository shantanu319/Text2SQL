```sql
WITH ModelStatus AS (
    SELECT 
        m.name,
        m.version,
        s.status,
        CASE 
            WHEN MAX(CASE WHEN t.model != 'Stack' THEN t.score ELSE NULL END) < t.score THEN 'strong'
            WHEN MAX(CASE WHEN t.model != 'Stack' THEN t.score ELSE NULL END) = t.score THEN 'soft'
            ELSE 'unknown'
        END AS status_type
    FROM 
        models m
    JOIN 
        tests t ON m.id = t.model_id
    GROUP BY 
        m.name, m.version, s.status
),
L1ModelCounts AS (
    SELECT 
        l1.name,
        l1.version,
        COUNT(*) AS count,
        CASE 
            WHEN ss.status_type = 'strong' THEN 1
            WHEN ss.status_type = 'soft' THEN 2
            ELSE 0
        END AS status_code
    FROM 
        l1_models l1
    LEFT JOIN 
        ModelStatus ss ON l1.name = ss.name AND l1.version = ss.version
    GROUP BY 
        l1.name, l1.version
)
SELECT 
    l1.name,
    l1.version,
    sc.count,
    sc.status