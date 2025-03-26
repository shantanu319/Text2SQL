SELECT AVG(wage), AVG(cpi)
FROM (
    SELECT 
        AVG(CASE WHEN industry = 'All Industries' THEN wage ELSE NULL END) AS wage,
        AVG(CASE WHEN item = 'All Consumer Items' THEN cpi ELSE NULL END) AS cpi
    FROM