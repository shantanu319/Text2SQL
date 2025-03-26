SELECT 
    (SUM(CASE WHEN helmet = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS Percentage_Helmet_Wearing,
    (SUM(CASE WHEN helmet = 'No' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS Percentage_No_Helmet