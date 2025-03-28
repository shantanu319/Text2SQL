```sql
SELECT 
    T3.zip_code,
    T4.employment_rate
FROM 
    zip_codes AS T1
JOIN 
    fdic_insurance AS T2 ON T1.zip_code = T2.zip_code
JOIN 
    census_data AS T3 ON T1.zip_code = T3.zip_code
JOIN 
    employment_rates AS T4 ON T3.zip_code = T4.zip_code
WHERE 
    T2.bank_locations = (SELECT MIN(bank_locations) FROM fdic_insurance)
ORDER BY 
    T4.employment_rate ASC
LIMIT 1;
```