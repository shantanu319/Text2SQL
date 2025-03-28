```sql
-- Step 1: Select patients with non-null clinical_stage and disease_code 'KIRP'
SELECT DISTINCT T1.case_barcode
FROM TCGA_bioclin_v0.Clinical AS T1
JOIN TCGA_hg38_data_v0.RNAseq_Gene_Expression AS T2 ON T1.case_barcode = T2.case_barcode
WHERE T1.clinical_stage IS NOT NULL AND T1.disease_code = 'KIRP';

-- Step 2: Randomly split patients into training and test sets using FARM_FINGERPRINT
WITH SplitPatients AS (
    SELECT 
        case_barcode,
        ROW_NUMBER() OVER (PARTITION BY FARM_FINGERPRINT(case_barcode) ORDER BY RAND()) AS rn
    FROM 
        TCGA_bioclin_v0.Clinical AS T1
    JOIN 
        TCGA_hg38_data_v0.RNAseq_Gene_Expression AS T2 ON T1.case_barcode = T2.case_barcode
    WHERE 
        T1.clinical_stage IS NOT NULL AND T1.disease_code = 'KIRP'
)
SELECT 
    case_barcode,
    CASE 
        WHEN