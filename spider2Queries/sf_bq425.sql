```sql
SELECT DISTINCT T1.trade_name, MAX(T2.approval_date) AS latest_approval_date
FROM CHEMBL_MOLECULES AS T1
JOIN CHEMBL_APPROVALS AS T2 ON T1.molecule_chembl_id = T2.molecule_chembl_id
WHERE T1.company = 'SanofiAventis'
GROUP BY T1.trade_name;
```