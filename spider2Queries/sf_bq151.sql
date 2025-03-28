```sql
-- Step 1: Load necessary datasets
LOAD DATA INFILE 'TCGA_PANCAN_GENE_MUTATIONS.csv' INTO TABLE t_c_g_a_gene_mutations;
LOAD DATA INFILE 'TCGA_PANCAN_CLINICAL_DATA.csv' INTO TABLE t_c_g_a_clinical_data;

-- Step 2: Join datasets based on patient IDs
SELECT 
    c.id AS patient_id,
    m.kras_mutation AS kras_status,
    m.tp53_mutation AS tp53_status,
    c.follow_up_data AS follow_up_info
FROM 
    t_c_g_a_clinical_data c
JOIN 
    t_c_g_a_gene_mutations m ON c.patient_id = m.patient_id;

-- Step 3: Calculate chi-squared statistic for co-occurrence vs independent mutations
WITH MutationCounts AS (
    SELECT 
        kras_status,
        tp53_status,
        COUNT(*) AS total_count
    FROM 
        joined_datasets
    GROUP BY 
        kras_status, tp53_status
),
CoOccurrenceCounts AS (
    SELECT 
        kras_status,
        tp53_status,
        COUNT(*) AS co_occurrence_count
    FROM 
        joined_datasets