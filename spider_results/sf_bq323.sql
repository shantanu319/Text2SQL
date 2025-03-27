```sql
WITH T1 AS (
    SELECT 
        StudyInstanceUID,
        SeriesInstanceUID,
        Modality,
        SeriesDescription,
        RepetitionTime,
        EchoTime,
        SliceThickness
    FROM 
        prostatex
    WHERE 
        Modality = 'MR'
),
T2 AS (
    SELECT 
        StudyInstanceUID,
        SeriesInstanceUID,
        CASE 
            WHEN SeriesDescription LIKE '%t2_tse_tra%' THEN 't2w_prostateX'
            WHEN SeriesDescription LIKE '%ADC%' THEN 'adc_prostateX'
            ELSE NULL
        END AS Label
    FROM 
        T1
)
SELECT 
    SUM(AvgRepetitionTime) + SUM(AvgEchoTime) + SUM(AvgSliceThickness) AS CombinedOverallAverage
FROM (
    SELECT 
        AVG(RepetitionTime) AS AvgRepetitionTime,
        AVG(EchoTime) AS AvgEchoTime,
        AVG(SliceThickness) AS AvgSliceThickness
    FROM 
        T2
    WHERE 
        Label IS NOT NULL
) AS Subquery;
```