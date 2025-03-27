To compile the critical details on traffic accidents in 2015 from an info document, you can use the following SQL query. This assumes that your table is named `traffic_accidents` and it has columns such as `year`, `location`, `description`, etc.

```sql
SELECT *
FROM traffic_accidents
WHERE year = 2015;
```

This query selects all columns from the `traffic_accidents` table where the `year` column matches 2015. If you need specific columns or additional conditions, please let me know!