```sql
SELECT CrimeCategory, COUNT(*) AS NumberOfIncidents
FROM Crimes
WHERE Borough = 'Barking and Dagenham'
GROUP BY CrimeCategory
ORDER BY NumberOfIncidents DESC
LIMIT 3;
```