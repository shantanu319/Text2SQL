```sql
SELECT 
    patent_id,
    title,
    application_date,
    COUNT(CASE WHEN citation_date < application_date THEN 1 END) AS backward_citations_1_month,
    COUNT(CASE WHEN citation_date > application_date THEN 1 END) AS forward_citations_1_month,
    abstract_text
FROM 
    patents
WHERE 
    cpc_category IN ('C05', 'A01G')
    AND (
        EXISTS (
            SELECT 1
            FROM citations
            WHERE patent_id = patents.patent_id
                AND citation_date < application_date
                AND ABS(citation_date - application_date) <= 30
        )
        OR EXISTS (
            SELECT 1
            FROM citations
            WHERE patent_id = patents.patent_id
                AND citation_date > application_date
                AND ABS(citation_date - application_date) <= 30
        )
    )
GROUP BY 
    patent_id, title, application_date, abstract_text
ORDER BY 
    application_date;
```