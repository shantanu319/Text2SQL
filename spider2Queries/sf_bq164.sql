```sql
SELECT 
    Filename,
    HTAN_Parent_Biospecimen_ID,
    Component,
    File_Format,
    Entity_ID,
    Run_ID
FROM 
    Spatial_Transcriptomics_Data
WHERE 
    Run_ID = 'HT264P1-S1H2Fc2U1Z1Bs1-H2Bs2-Test'
UNION ALL
SELECT 
    Filename,
    HTAN_Parent_Biospecimen_ID,
    Component,
    File_Format,
    Entity_ID,
    Run_ID
FROM 
    scRNAseq_Data
WHERE 
    Run_ID = 'HT264P1-S1H2Fc2U1Z1Bs1-H2Bs2-Test';
```