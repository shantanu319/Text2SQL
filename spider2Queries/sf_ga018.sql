To achieve this, you can use the following SQL query:

```sql
WITH PageViews AS (
    SELECT 
        user_id,
        event_date,
        CASE 
            WHEN page_name = 'product_list_page' THEN 'PLP'
            WHEN page_name = 'product_detail_page' THEN 'PDP'
            ELSE NULL
        END AS page_type
    FROM 
        page_views
    WHERE 
        event_date = '2021-01-02'
),
Transitions AS (
    SELECT 
        user_id,
        MIN(event_date) AS first_plp_date,
        MAX(event_date) AS last_pdp_date
    FROM 
        PageViews
    WHERE 
        page_type IN ('PLP', 'PDP')
    GROUP BY 
        user_id
)
SELECT 
    COUNT(CASE WHEN t.first_plp_date < t.last_pdp_date THEN 1 ELSE NULL END) AS plp_to_pdp_transitions,
    COUNT(*) AS total_plp_views,
    CAST(COUNT(CASE WHEN t.first_plp_date < t.last_pdp_date THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(*) AS percentage_plp_to_pdp
FROM 
    Trans