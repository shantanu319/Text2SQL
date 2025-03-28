```sql
SELECT 
    ctc.census_tract_id,
    AVG(fec.donation_amount) AS avg_donation_amount,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fec.donation_amount) OVER (PARTITION BY ctc.census_tract_id) AS median_donation_amount,
    ctc.median_income
FROM 
    census_tracts ctc
JOIN 
    zip_code_to_census_tract_crosswalk zctc ON ctc.zip_code = zctc.zip_code
JOIN 
    fec_individual_contributions fec ON zctc.census_tract_id = fec.census_tract_id
WHERE 
    ctc.county_name = 'Kings County'
    AND fec.donor_state = 'NY'
GROUP BY 
    ctc.census_tract_id;
```