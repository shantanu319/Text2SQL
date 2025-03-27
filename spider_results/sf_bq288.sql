```sql
SELECT COUNT(*) AS Total_Banking_Institutions
FROM Banking_Institutions
WHERE Institution_Name LIKE 'Bank%'
AND Establishment_Date BETWEEN '1900-01-01' AND '2000-12-31'
GROUP BY State
ORDER BY SUM(Assets) DESC
LIMIT 1;
```