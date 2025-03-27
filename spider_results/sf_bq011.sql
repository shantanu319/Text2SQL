```sql
SELECT COUNT(*) AS pseudo_users_active_last_7_inactive_last_2
FROM pseudo_users
WHERE active_date >= DATE_SUB('2021-01-07', INTERVAL 7 DAY)
  AND active_date < DATE_SUB('2021-01-07', INTERVAL 2 DAY)
  AND inactive_date IS NULL;
```