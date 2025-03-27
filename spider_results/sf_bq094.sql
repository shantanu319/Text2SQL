```sql
SELECT 
    c.committee_name,
    COUNT(DISTINCT ca.candidate_id) AS num_unique_candidates,
    GROUP_CONCAT(ca.candidate_name ORDER BY ca.candidate_name SEPARATOR ', ') AS candidate_names,
    SUM(sc.contribution_amount) AS total_small_dollar_donations
FROM 
    committees c
JOIN 
    committee_support cs ON c.committee_id = cs.committee_id
JOIN 
    candidates ca ON cs.candidate_id = ca.candidate_id
JOIN 
    small_dollar_contributions sc ON cs.committee_id = sc.committee_id AND cs.candidate_id = sc.candidate_id
WHERE 
    c.year = 2016
    AND sc.contribution_amount BETWEEN 0 AND 200
GROUP BY 
    c.committee_id
HAVING 
    SUM(sc.contribution_amount) > 0;
```