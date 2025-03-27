```sql
SELECT 
    YEAR(TermStart) AS TermYear,
    COUNT(CASE WHEN TermEnd >= '1999-12-31' THEN 1 ELSE NULL END) / COUNT(*) AS ProportionRemainingInOffice
FROM 
    Legislators
WHERE 
    TermStart BETWEEN '1917-01-01' AND '1999-12-31'
GROUP BY 
    YEAR(TermStart)
ORDER BY 
    TermYear;
```

This SQL query calculates the proportion of legislators who started their first term between 1917 and 1999 and remained in office on December 31st of each year, tracking annually for up to 20 years following their initial term start. It uses a `CASE` statement within the `COUNT` function to determine if a legislator remained in office by checking if their `TermEnd` date is after or equal to December 31st of the current year. The results are grouped by the year of the term start and ordered by the term year.