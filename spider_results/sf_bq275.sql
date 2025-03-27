```sql
SELECT v.visitor_id
FROM visitors v
JOIN transactions t ON v.visitor_id = t.visitor_id
WHERE t.device_type = 'mobile'
  AND t.transaction_date > (
    SELECT MIN(t2.transaction_date)
    FROM transactions t2
    WHERE t2.visitor_id = v.visitor_id
      AND t2.first_transaction = TRUE
  );
```