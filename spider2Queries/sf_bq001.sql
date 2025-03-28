```sql
SELECT 
    v.visitor_id,
    DATEDIFF(v.first_visit_date, t.first_transaction_date) AS days_elapsed,
    t.device_type
FROM 
    visitors v
JOIN 
    transactions t ON v.visitor_id = t.visitor_id AND t.transaction_date BETWEEN '2017-02-01' AND '2017-02-28'
WHERE 
    v.first_visit_date IS NOT NULL AND t.first_transaction_date IS NOT NULL
GROUP BY 
    v.visitor_id, v.first_visit_date, t.first_transaction_date, t.device_type;
```

This query joins the `visitors` and `transactions` tables based on the `visitor_id`, filters for transactions made in February 2017, calculates the number of days between the first visit and the first transaction, and retrieves the device type of the first transaction. The results are grouped by visitor ID, first visit date, first transaction date, and device type.