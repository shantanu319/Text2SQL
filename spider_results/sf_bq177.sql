```sql
SELECT 
    p.provider_id,
    YEAR(i.inpatient_cost_date) AS inpatient_year,
    AVG(i.average_medicare_payments * i.total_discharges) AS inpatient_average_cost,
    YEAR(o.outpatient_cost_date) AS outpatient_year,
    AVG(o.average_total_payments * o.outpatient_services) AS outpatient_average_cost
FROM 
    providers p
JOIN 
    inpatient_costs i ON p.provider_id = i.provider_id AND i.cost_period BETWEEN '2011-01-01' AND '2015-12-31'
JOIN 
    outpatient_costs o ON p.provider_id = o.provider_id AND o.cost_period BETWEEN '2011-01-01' AND '2015-12-31'
WHERE 
    p.provider_id IN (
        SELECT 
            provider_id
        FROM 
            inpatient_costs
        GROUP BY 
            provider_id
        ORDER BY 
            SUM(average_medicare_payments * total_discharges) DESC
        LIMIT 1
    )
GROUP BY 
    p.provider_id, YEAR(i.inpatient_cost_date), YEAR(o.outpatient_cost_date);
```

This SQL query calculates the yearly average in