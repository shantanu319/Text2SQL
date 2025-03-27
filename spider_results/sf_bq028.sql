```sql
SELECT 
    p.name AS package_name,
    v.version AS version,
    COUNT(*) AS github_stars
FROM 
    npm_packages p
JOIN 
    npm_package_versions v ON p.id = v.package_id
WHERE 
    p.latest_release = TRUE
GROUP BY 
    p.name, v.version
ORDER BY 
    github_stars DESC
LIMIT 8;
```