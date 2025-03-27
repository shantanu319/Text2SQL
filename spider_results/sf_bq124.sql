```sql
SELECT COUNT(DISTINCT patientId) AS alive_patients
FROM patients
WHERE deceasedDateTime IS NULL
  AND (diagnosis = 'Diabetes' OR diagnosis = 'Hypertension')
  AND (
    SELECT COUNT(DISTINCT medicationId)
    FROM prescriptions
    WHERE patientId = patients.patientId
      AND status = 'active'
  ) >= 7;
```