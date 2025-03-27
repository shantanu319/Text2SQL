```sql
SELECT SUM(T1.frames) AS total_frames
FROM TCGA_BRCA_images AS T1
JOIN SpecimenPreparationSequence AS T2 ON T1.image_id = T2.image_id
WHERE T1.modality = 'SM' AND T2.staining_step LIKE '%eosin%'
GROUP BY T1.collection;
```

This query calculates the total number of frames across all whole slide microscopy images from the TCGA-BRCA collection that meet the specified criteria. It joins the `TCGA_BRCA_images` table with the `SpecimenPreparationSequence` table on the image ID, filters for images using the SM modality and including an eosin-based staining step, sums up the frames for each collection, and groups the results by collection. The result is a single value representing the total number of frames across all qualifying images.