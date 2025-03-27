```sql
SELECT DISTINCT 
    T1.case_id,
    T1.clone_id,
    T2.chromosome_number AS chromosome_13,
    T2.start_position AS start_13,
    T2.end_position AS end_13,
    T3.chromosome_number AS chromosome_17,
    T3.start_position AS start_17,
    T3.end_position AS end_17,
    T4.chromosome_number AS chromosome_11,
    T4.start_position AS start_11,
    T4.end_position AS end_11,
    T5.karyotype_info
FROM 
    cases T1
JOIN 
    chromosome_details T2 ON T1.case_id = T2.case_id AND T2.chromosome_number = '13'
JOIN 
    chromosome_details T3 ON T1.case_id = T3.case_id AND T3.chromosome_number = '17'
JOIN 
    chromosome_details T4 ON T1.case_id = T4.case_id AND T4.chromosome_number = '11'
JOIN 
    karyotype_info T5 ON T1.case_id = T5.case_id
WHERE 
    T2.start_position BETWEEN 48303751