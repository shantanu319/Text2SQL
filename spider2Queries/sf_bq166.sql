```sql
-- Step 1: Select relevant columns from the copy_number_segment_allelic_hg38_gdc_r23 dataset for TCGA-KIRC samples
SELECT 
    sample_id,
    chromosome,
    cytoband,
    max_copy_number
FROM 
    copy_number_segment_allelic_hg38_gdc_r23
WHERE 
    sample_id IN (
        SELECT 
            sample_id
        FROM 
            gdc_samples
        WHERE 
            tumor_type = 'KIRC'
    )
    AND sample_id NOT LIKE '%-TUMOR%'
    AND sample_id NOT LIKE '%-NORMAL%';

-- Step 2: Merge the selected segments with the cytogenetic band definitions in CytoBands_hg38
SELECT 
    s.sample_id,
    s.chromosome,
    s.cytoband,
    s.max_copy_number,
    c.band_name
FROM 
    (SELECT 
         sample_id,
         chromosome,
         cytoband,
         max_copy_number
     FROM 
         copy_number_segment_allelic_hg38_gdc_r23
     WHERE 
         sample_id IN (
             SELECT 
                 sample_id
             FROM 
                 gdc_samples