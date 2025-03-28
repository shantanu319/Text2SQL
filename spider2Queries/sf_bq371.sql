```sql
SELECT MAX(AVG_Value) - MIN(AVG_Value) AS Difference
FROM (
    SELECT 
        QUARTER(invoice_date) AS Quarter,
        AVG(unit_price * quantity) AS AVG_Value
    FROM 
        invoices
    WHERE 
        YEAR(invoice_date) = 2013
    GROUP BY 
        Quarter
) AS Invoice_Quarters;
```