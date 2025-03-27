```sql
SELECT drug_id, drug_type, withdrawal_status
FROM molecule_data
WHERE drug_type IN ('Keytruda', 'Vioxx', 'Premarin', 'Humira')
  AND approval_status = 'approved'
  AND black_box_warning = 'black box';
```