SELECT 
    YEAR(output_date) AS year,
    AVG(CASE WHEN method = 'Merged' THEN output_value ELSE NULL END) - AVG(CASE WHEN method = 'Transactions' THEN output_value ELSE NULL END) AS difference
FROM (
    SELECT 
        output_date,
        output_value,
        'Merged' AS method
    FROM 
        merged_inputs_outputs
    UNION ALL
    SELECT 
        transaction_date,
        output_value,
        'Transactions' AS method
    FROM 
        transactions
)