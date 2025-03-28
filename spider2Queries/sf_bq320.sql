```sql
SELECT COUNT(DISTINCT StudyInstanceUID)
FROM dicom_pivot
WHERE LOWER(SegmentedPropertyTypeCodeSequence) = '15825003'
  AND collection_id IN ('Community', 'nsclc_radiomics');
```