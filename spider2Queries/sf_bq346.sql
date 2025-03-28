```sql
SELECT 
    SegmentedPropertyCategory.CodeMeaning,
    COUNT(*) AS Frequency
FROM 
    DICOMData
WHERE 
    Modality = 'SEG'
    AND SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
GROUP BY 
    SegmentedPropertyCategory.CodeMeaning
ORDER BY 
    Frequency DESC
LIMIT 5;
```