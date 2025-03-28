```sql
SELECT 
    YEAR(output_date) AS year,
    AVG(merged_output_value) - AVG(transactions_average) AS difference
FROM (
    SELECT 
        DATE_TRUNC('year', output_date) AS year,
        SUM(input_value + output_value) / COUNT(*) AS merged_output_value
    FROM 
        inputs
    JOIN 
        outputs ON inputs.id = outputs.input_id
    WHERE 
        outputs.output_type = 'output'
    GROUP BY 
        YEAR(output_date)
) AS merged_outputs
JOIN (
    SELECT 
        DATE_TRUNC('year', output_date) AS year,
        AVG(output_value) AS transactions_average
    FROM 
        transactions
    GROUP BY 
        YEAR(output_date)
) AS transactions
ON 
    merged_outputs.year = transactions.year
WHERE 
    EXISTS (
        SELECT 1
        FROM merged_outputs
        WHERE merged_outputs.year = transactions.year
    )
GROUP BY 
    year;
```

This SQL query calculates the annual differences in Bitcoin output value averages between two methods: merged input/output records and direct use of the transactions table. It shows the difference only for years with data in both methods. The query uses subqueries to calculate the yearly averages for each method and then joins these results to find