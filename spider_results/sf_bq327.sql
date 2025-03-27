```sql
SELECT COUNT(*) 
FROM debt_indicators 
WHERE country = 'Russia' AND value = 0 AND value IS NOT NULL;
```