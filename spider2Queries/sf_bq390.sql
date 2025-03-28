```sql
SELECT DISTINCT StudyInstanceUID 
FROM qin_prostate_repeatability 
WHERE Modality = 'T2' AND ImageType LIKE '%AXIAL%' 
  AND AnatomySegmentationLabel = 'Peripheral zone';
```