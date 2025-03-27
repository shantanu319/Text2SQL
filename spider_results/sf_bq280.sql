```sql
SELECT DisplayName
FROM Users
WHERE Reputation > 10
ORDER BY Answers DESC
LIMIT 1;
```