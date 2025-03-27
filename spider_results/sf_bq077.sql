SELECT YEAR(T1.date) AS Year, MAX(T2.motor_thefts) AS Highest_Monthly_Thefts
FROM thefts AS T1
JOIN monthly_thefts AS T2 ON T1.id = T2.theft_id
WHERE YEAR(T1.date) BETWEEN 2010 AND 2016
GROUP BY YEAR(T1.date);