SELECT 
    AVG(unit_price * quantity) AS avg_invoice_value,
    QUARTER(invoice_date) AS quarter
FROM 
    invoices