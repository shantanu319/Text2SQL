```sql
-- Step 1: Find the DRG definition with the highest total number of discharges nationwide
WITH TopDRG AS (
    SELECT 
        DRG,
        SUM(Discharges) AS TotalDischarges
    FROM 
        CMS_Medicare_inpatient_charges_data_2014
    GROUP BY 
        DRG
    ORDER BY 
        TotalDischarges DESC
    LIMIT 1
)

-- Step 2: Identify the top three cities with the most discharges for the most common DRG definition
WITH TopCities AS (
    SELECT 
        City,
        SUM(Discharges) AS TotalDischarges
    FROM 
        CMS_Medicare_inpatient_charges_data_2014
    WHERE 
        DRG = (SELECT DRG FROM TopDRG)
    GROUP BY 
        City
    ORDER BY 
        TotalDischarges DESC
    LIMIT 3
)

-- Step 3: Calculate the weighted average total payments for the top three cities
SELECT 
    City,
    AVG(TotalPayments * Discharges) / SUM(Discharges) AS WeightedAverageTotalPayments
FROM 
    TopCities
JOIN 
    CMS_Medicare_inpatient_charges_data_