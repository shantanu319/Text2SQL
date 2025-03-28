```sql
SELECT State, Medication, COUNT(*) AS Prescriptions
FROM PrescriptionData
WHERE Year = 2014
GROUP BY State, Medication
ORDER BY State, COUNT(*) DESC;
```