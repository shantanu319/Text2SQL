```sql
SELECT 
    T2.assignee_name AS citing_assignee,
    SUBSTRING_INDEX(T1.cpc_codes, ' ', 1) AS primary_cpc_subclass_title,
    COUNT(*) AS citation_count
FROM 
    patent_citations AS T1
JOIN 
    assignees AS T2 ON T1.assignee_id = T2.assignee_id
WHERE 
    T1.filing_date IS NOT NULL
    AND T2.name != 'DENSO CORP'
    AND T1.patent_number IN (
        SELECT 
            patent_number
        FROM 
            patent_assignments AS T3
        WHERE 
            T3.assignee_name = 'DENSO CORP'
    )
GROUP BY 
    T2.assignee_name, 
    SUBSTRING_INDEX(T1.cpc_codes, ' ', 1)
ORDER BY 
    citing_assignee, 
    primary_cpc_subclass_title;
```