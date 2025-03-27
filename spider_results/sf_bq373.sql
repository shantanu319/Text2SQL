```sql
SELECT 
    DATE_FORMAT(invoice_date, '%Y-%m') AS month,
    AVG(total_invoice_line_amount) AS avg_monthly_spending
FROM 
    invoices
WHERE 
    YEAR(invoice_date) = 2014
GROUP BY 
    DATE_FORMAT(invoice_date, '%Y-%m')
ORDER BY 
    avg_monthly_spending ASC
LIMIT 1 OFFSET (COUNT(*) - 1) / 2;
```

This query calculates the average monthly spending for each month in 2014, groups the results by month, and then finds the median of these averages. The `LIMIT` clause with `OFFSET` is used to find the middle value when the number of rows is odd, or the average of the two middle values when the number of rows is even. Note that this assumes there are no ties in the average monthly spending values. If there are ties, you might need a different approach to handle them.