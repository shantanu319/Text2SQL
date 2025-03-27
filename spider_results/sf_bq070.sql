Certainly! Below is an SQL query that should meet your requirements:

```sql
SELECT *
FROM dicom_all
WHERE 
    collection IN ('TCGA-LUAD', 'TCGA-LUSC')
    AND image_type = 'SM'
    AND volume = 1
    AND slide_compression != 'other'
    AND specimen_preparation_step LIKE '%Embedding medium%'
    AND tissue_type IN ('normal', 'tumor')
    AND cancer_subtype IS NOT NULL;
```

### Explanation:
- **collection**: Filters for datasets from the specified TCGA collections.
- **image_type**: Ensures that only SM (Slide Marked) images are included.
- **volume**: Checks if the image is marked as a volume.
- **slide_compression**: Excludes slides with compression type "other".
- **specimen_preparation_step**: Ensures that the embedding medium is used.
- **tissue_type**: Filters for normal or tumor tissues.
- **cancer_subtype**: Ensures that the cancer subtype is reported. The `IS NOT NULL` condition ensures that there is a report for the cancer subtype.

Make sure that the column names (`collection`, `image_type`, etc.) match those in your actual database schema. Adjust them as necessary