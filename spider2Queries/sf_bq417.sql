```sql
SELECT 
    IdentificationDetails,
    StudyAndSeriesInformation,
    StorageLocation,
    TotalSizeInMB
FROM 
    MedicalImages
WHERE 
    Gender = 'Male'
    AND NumericPortionOfPatientAge = 18
    AND BodyPartExamined = 'MEDIASTINUM'
    AND StudyDate > '2014-09-01';
```