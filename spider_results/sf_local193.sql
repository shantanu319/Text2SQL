SELECT AVG(LTV_7_days), AVG(LTV_30_days)
FROM (
    SELECT 
        LTV,
        CASE 
            WHEN DATEDIFF(HOUR, '2021-01-01', Initial_Purchase_Date) <= 7 THEN LTV
            ELSE NULL
        END AS LTV_7_days,
        CASE 
            WHEN DATEDIFF(HOUR, '2021-01-01', Initial_Purchase_Date) <= 30 THEN LTV
            ELSE NULL
        END AS LTV_30_days
    FROM Your_Table_Name