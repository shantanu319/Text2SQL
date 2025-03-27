```sql
SELECT 
    T3.state_name,
    SUM(T3.claim_count) AS total_claim_count,
    SUM(T3.drug_cost) AS total_drug_cost
FROM 
    prescription_data AS T1
JOIN 
    state_data AS T2 ON T1.state_id = T2.state_id
JOIN 
    drug_data AS T3 ON T1.drug_id = T3.drug_id
WHERE 
    T1.year = 2014 AND T1.state_id = (SELECT state_id FROM prescription_data WHERE year = 2014 ORDER BY total_prescriptions DESC LIMIT 1)
GROUP BY 
    T3.state_name
ORDER BY 
    total_claim_count DESC
LIMIT 5;
```