```sql
SELECT 
    r.name AS repo_name,
    COUNT(DISTINCT w.id) AS total_watchers,
    COUNT(DISTINCT ie.event_id) AS total_issues,
    COUNT(DISTINCT f.fork_count) AS total_forks
FROM 
    repositories r
JOIN 
    licenses l ON r.license_id = l.id
JOIN 
    watch_counts wc ON r.id = wc.repository_id
JOIN 
    issue_events ie ON r.id = ie.repository_id
JOIN 
    fork_counts f ON r.id = f.repository_id
WHERE 
    l.open_source_license IN ('artistic-2.0', 'isc', 'mit', 'apache-2.0')
    AND MONTH(wc.created_at) = 4
    AND YEAR(wc.created_at) = 2022
    AND MONTH(ie.created_at) = 4
    AND YEAR(ie.created_at) = 2022
    AND MONTH(f.created_at) = 4
    AND YEAR(f.created_at) = 2022
    AND r.master_branch_files LIKE '%.py'
GROUP BY 
    r.name
ORDER BY 
    total_forks + total_issues + total_watchers DESC