```sql
SELECT DISTINCT T1.url 
FROM npm_package_versions AS T1 
JOIN npm_packages AS T2 ON T1.package_id = T2.id 
WHERE T2.name NOT LIKE '%@%' 
  AND T1.link_label = 'SOURCE_REPO' 
  AND T1.url LIKE '%github.com%'
ORDER BY T1.dependencies_count DESC 
LIMIT 1;
```