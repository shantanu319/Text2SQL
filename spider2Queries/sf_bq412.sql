```sql
SELECT 
    page_url,
    first_shown_time,
    last_shown_time,
    removal_reason,
    violation_category,
    times_shown_lower_bound,
    times_shown_upper_bound
FROM 
    ads
WHERE 
    region_code = 'HR'
    AND times_shown_availability_date IS NULL
    AND times_shown_lower_bound > 10000
    AND times_shown_upper_bound < 25000
    AND ad_used_non_unused_audience_selection_approach
ORDER BY 
    last_shown_time DESC
LIMIT 5;
```