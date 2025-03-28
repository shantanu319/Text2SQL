```sql
SELECT YEAR(IncidentDate) AS IncidentYear, COUNT(*) AS TotalOtherThefts
FROM Incidents
WHERE Category = 'Theft and Handling'
  AND SubCategory = 'Other Theft'
  AND Borough = 'Westminster'
GROUP BY IncidentYear;
```