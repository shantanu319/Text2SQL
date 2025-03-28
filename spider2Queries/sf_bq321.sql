```sql
SELECT COUNT(DISTINCT StudyInstanceUID) 
FROM qin_prostate_repeatability 
WHERE SeriesDescription IN ('DWI', 'T2 Weighted Axial', 'Apparent Diffusion Coefficient', 'T2 Weighted Axial Segmentations');
```