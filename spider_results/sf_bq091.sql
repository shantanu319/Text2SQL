```sql
SELECT YEAR(filing_date) AS filing_year, COUNT(*) AS application_count
FROM patents
WHERE assignee = (
    SELECT assignee
    FROM patents
    WHERE patent_category = 'A61'
    GROUP BY assignee
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
GROUP BY YEAR(filing_date)
ORDER BY application_count DESC
LIMIT 1;
```