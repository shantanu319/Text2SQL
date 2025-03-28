```sql
SELECT 
    tier,
    MAX(profit) AS highest_profit,
    MIN(profit) AS lowest_profit
FROM (
    SELECT 
        customer_id,
        profit,
        DENSE_RANK() OVER (ORDER BY profit DESC) AS tier
    FROM 
        sales
    WHERE 
        customer_country = 'Italy'
        AND sale_date BETWEEN '2021-12-01' AND '2021-12-31'
) AS ranked_customers
GROUP BY 
    tier
LIMIT 10;
```

This query calculates the highest and lowest profits for Italian customers, segments them into ten evenly divided tiers based on their December 2021 sales profits, and returns the results. The `DENSE_RANK()` function is used to assign a rank to each customer's profit in descending order, which helps in creating the tiers. The `LIMIT` clause ensures that only the top 10 tiers are returned. Adjust the table and column names as necessary to fit your database schema.