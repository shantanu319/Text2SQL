To analyze the average percentage change in sales between the 12 weeks before and after June 15, 2020, for each attribute type (region, platform, age band, demographic, and customer type), you can use the following SQL query:

```sql
WITH SalesData AS (
    SELECT 
        date,
        attribute_type,
        attribute_value,
        sales_amount,
        LAG(sales_amount) OVER (PARTITION BY attribute_type ORDER BY date) AS prev_sales_amount
    FROM 
        sales_data
),
PercentageChange AS (
    SELECT 
        attribute_type,
        attribute_value,
        (sales_amount - prev_sales_amount) / NULLIF(prev_sales_amount, 0) * 100 AS percentage_change
    FROM 
        SalesData
)
SELECT 
    attribute_type,
    AVG(percentage_change) AS average_percentage_change
FROM 
    PercentageChange
GROUP BY 
    attribute_type
ORDER BY 
    average_percentage_change DESC
LIMIT 1;
```

### Explanation:
1. **SalesData CTE**: This Common Table Expression calculates the previous sales amount for each record using the `LAG` function. It partitions the data by `attribute_type` and orders it by `date`.

2. **