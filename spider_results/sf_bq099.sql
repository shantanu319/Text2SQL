```sql
SELECT 
    a.name AS assignee_name,
    COUNT(*) AS total_applications,
    MAX(y.year) AS year_with_most_apps,
    y.applications AS num_apps_in_year,
    c.country_code AS country_code_with_most_apps
FROM 
    assignees a
JOIN 
    (
        SELECT 
            assignee_id,
            YEAR(application_date) AS year,
            COUNT(*) AS applications
        FROM 
            applications
        WHERE 
            patent_class = 'A01B3'
        GROUP BY 
            assignee_id, YEAR(application_date)
    ) y ON a.id = y.assignee_id
JOIN 
    countries c ON a.country_code = c.code
GROUP BY 
    a.name
ORDER BY 
    total_applications DESC
LIMIT 3;
```

This SQL query retrieves the required information for the top 3 assignees based on the total number of applications for the specified patent class. It joins the `assignees`, `applications`, and `countries` tables to gather the necessary data and then groups the results by assignee to calculate the total number of applications, the year with the most applications, the number of applications in that year, and the country code with the most applications during that year.