```sql
SELECT stprba AS State, SUM(revenue) / 1000 AS TotalAnnualRevenue
FROM (
    SELECT stprba, 
           CASE WHEN measure_tag IN ('Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet') THEN value ELSE NULL END AS revenue
    FROM your_table_name
    WHERE fiscal_year = 2016 AND quarter BETWEEN 1 AND 4
      AND stprba IS NOT NULL AND stprba <> ''
) AS filtered_data
GROUP BY stprba
ORDER BY TotalAnnualRevenue DESC
LIMIT 1;
```

Replace `your_table_name` with the actual name of your table. This query filters the data for fiscal year 2016, considers only companies that provided four quarters of data, and excludes any entries where the state field is null or empty. It then calculates the total annual revenue for each state and orders the results by the total revenue in descending order, selecting the state with the highest total annual revenue.