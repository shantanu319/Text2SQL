```sql
SELECT DISTINCT T2.condition_code, COUNT(DISTINCT T3.active_medications) AS medication_count
FROM patients AS T1
JOIN patient_conditions AS T2 ON T1.patient_id = T2.patient_id
JOIN prescriptions AS T3 ON T1.patient_id = T3.patient_id
WHERE T1.last_name LIKE 'A%'
GROUP BY T2.condition_code
ORDER BY medication_count DESC
LIMIT 8;
```