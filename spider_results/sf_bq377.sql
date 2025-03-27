To extract and count the frequency of all package names listed in the `require` section of JSON-formatted content, you can use the following SQL query. This assumes that your JSON data is stored in a table named `json_content` with a column `content` that contains the JSON string.

```sql
SELECT 
    package_name,
    COUNT(*) AS frequency
FROM (
    SELECT 
        json_extract(content, '$.require.*') AS package_names
    FROM 
        json_content
) AS extracted_packages
UNPIVOT (
    package_name FOR package_index IN (1)
) AS unpivoted_packages
GROUP BY 
    package_name;
```

### Explanation:

1. **Extracting Package Names**: The subquery uses `json_extract` to extract all values from the `$.require.*` path in each JSON object. This will return an array of package names.

2. **Unpivoting**: The `UNPIVOT` clause is used to transform the array of package names into individual rows, making it easier to process each package name separately.

3. **Counting Frequencies**: Finally, the outer query groups by `package_name` and counts the occurrences using `COUNT(*)`, providing the frequency of each package name.