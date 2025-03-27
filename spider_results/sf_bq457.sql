```sql
SELECT 
    r.full_name,
    r.owner,
    r.hosting_platform_type,
    r.size_in_bytes,
    r.primary_language,
    r.fork_source_name,
    r.last_update_timestamp,
    ft.artifact_name,
    ft.library_name,
    ft.platform,
    ft.language
FROM 
    repositories r
JOIN 
    feature_toggle_artifacts ft ON r.id = ft.repository_id
WHERE 
    ft.artifact_name IN ('toggle1', 'toggle2') -- Replace with actual artifact names
    AND ft.library_name IN ('library1', 'library2') -- Replace with actual library names
    AND ft.platform IN ('platform1', 'platform2') -- Replace with actual platforms
    AND ft.language IN ('language1', 'language2'); -- Replace with actual languages
```

This SQL query retrieves detailed information about repositories that use specific feature toggle libraries. It joins the `repositories` table with the `feature_toggle_artifacts` table based on the repository ID and filters the results based on the specified artifact names, library names, platforms, and languages. The query includes columns for the repository's full name, owner, hosting platform type, size in bytes, primary programming language, fork source name, last update