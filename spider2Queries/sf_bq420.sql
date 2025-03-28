```sql
SELECT 
    p.application_number,
    p.first_office_action_date,
    p.first_publication_number,
    p.first_publication_date,
    LENGTH(p.filed_claims) AS length_of_filed_claims,
    p.grant_date,
    COUNT(CASE WHEN p.rejection_reason = 'Section 101' AND p.allowed_claims_count = 0 THEN 1 END) AS rejection_count
FROM 
    patents p
WHERE 
    p.grant_date BETWEEN '2010-01-01' AND '2023-12-31'
GROUP BY 
    p.application_number, p.first_office_action_date, p.first_publication_number, p.first_publication_date, p.filed_claims, p.grant_date
ORDER BY 
    rejection_count DESC, LENGTH(p.filed_claims) ASC
LIMIT 5;
```

This SQL query selects the top 5 patents that meet the specified criteria, including those initially rejected under section 101 with no allowed claims, based on the length of their granted claims. It also includes the first office action date, first publication number, first publication date, length of the filed claims, grant date, and a count of