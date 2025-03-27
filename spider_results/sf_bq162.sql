```sql
SELECT DISTINCT 
    iat.Imaging_Assay_Type,
    CASE 
        WHEN EXISTS (
            SELECT 1 
            FROM id_provenance_r5 ipr 
            WHERE ipr.entityId = iat.Imaging_Assay_Type AND ipr.Level >= 2
        ) THEN 'Level2'
        ELSE ''
    END AS Level2,
    CASE 
        WHEN EXISTS (
            SELECT 1 
            FROM id_provenance_r5 ipr 
            WHERE ipr.entityId = iat.Imaging_Assay_Type AND ipr.Level >= 3
        ) THEN 'Level3'
        ELSE ''
    END AS Level3,
    CASE 
        WHEN EXISTS (
            SELECT 1 
            FROM id_provenance_r5 ipr 
            WHERE ipr.entityId = iat.Imaging_Assay_Type AND ipr.Level >= 4
        ) THEN 'Level4'
        ELSE ''
    END AS Level4
FROM 
    Imaging_Assay_Types iat
WHERE 
    iat.HTAN_Revision = 5
    AND iat.Center = 'WUSTL'
    AND iat.Component IS NOT NULL
    AND iat.Component NOT LIKE '%Auxiliary%'
    AND iat.Component NOT