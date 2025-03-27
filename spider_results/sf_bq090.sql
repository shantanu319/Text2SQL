SELECT AVG(T1.intrinsic_value) - AVG(T2.intrinsic_value) AS difference
FROM trades AS T1
JOIN trades AS T2 ON T1.id = T2.id
WHERE T1.strategy = 'feeling-lucky' AND T2.strategy = 'momentum'
  AND T1.side = 'long';