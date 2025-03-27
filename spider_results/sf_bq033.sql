```sql
SELECT 
    MONTH(filing_date) AS filing_month,
    COUNT(*) AS number_of_publications
FROM 
    us_publications
WHERE 
    abstract LIKE '%internet of things%'
    AND filing_date BETWEEN '2008-01-01' AND '2022-12-31'
GROUP BY 
    filing_month;
```

This query selects the month and counts the number of US publications related to IoT that have an abstract mentioning "internet of things" between January 1, 2008, and December 31, 2022. It groups the results by month, ensuring that even months without any filings are included in the output. The `LIKE` operator is used to filter the abstracts for the specific phrase. The `COUNT(*)` function counts the number of rows for each group, which corresponds to the number of publications per month. The `MONTH()` function extracts the month from the `filing_date`.