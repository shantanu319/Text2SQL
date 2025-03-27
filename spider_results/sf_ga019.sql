To convert the given text into an SQL query, we need to identify the relevant tables and columns. Assuming we have a table named `users` with columns such as `user_id`, `install_date`, and `uninstall_date`, the SQL query would look like this:

```sql
SELECT 
    (COUNT(CASE WHEN uninstall_date IS NULL OR install_date + INTERVAL '7 days' < uninstall_date THEN user_id END) * 100.0 / COUNT(user_id)) AS uninstall_percentage
FROM 
    users
WHERE 
    install_date BETWEEN '2018-08-01' AND '2018-09-30';
```

### Explanation:
- **COUNT(CASE ...)**: This is used to count the number of users who either did not uninstall the app within seven days or never uninstalled it.
  - `uninstall_date IS NULL`: Counts users who never uninstalled the app.
  - `install_date + INTERVAL '7 days' < uninstall_date`: Counts users who installed but did not uninstall within seven days.
- **(COUNT(...) * 100.0 / COUNT(user_id))**: Converts the count to a percentage by dividing by the total number of users.
-