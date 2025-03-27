```sql
SELECT z.zip_code, COUNT(*) AS hail_event_count
FROM zip_codes z
JOIN hail_reports hr ON z.zip_code = hr.zip_code
WHERE hr.event_date >= DATE_SUB(CURDATE(), INTERVAL 10 YEAR)
GROUP BY z.zip_code
ORDER BY hail_event_count DESC
LIMIT 5;
```