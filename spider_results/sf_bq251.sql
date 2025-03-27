```sql
SELECT DISTINCT 
    REGEXP_REPLACE(project_url, '(issues|pull|blob|tree)/[^/]+$', '') AS github_project_url
FROM (
    SELECT 
        pypi_package_name,
        project_urls
    FROM (
        SELECT 
            name AS pypi_package_name,
            jsonb_array_elements(text -> 'urls') AS project_urls
        FROM 
            pypi_packages
    ) AS extracted_urls
) AS filtered_urls
WHERE 
    project_urls LIKE '%github.com%'
ORDER BY 
    -- Assuming there's a column indicating download count or popularity
    download_count DESC
LIMIT 3;
```

This SQL query assumes you have a table named `pypi_packages` with columns `name` (the PyPI package name), `text` (JSONB containing package metadata), and `download_count` (or another column indicating package popularity). The query extracts project URLs from the JSONB data, filters them to include only GitHub URLs, cleans the URLs using a regular expression, and returns the top 3 GitHub project URLs based on download count. Ensure that the column names (`pypi_packages`, `name`, `text`, `project_urls`, `download_count`) match your actual database schema.