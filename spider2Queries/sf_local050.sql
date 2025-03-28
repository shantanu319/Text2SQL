To convert the given text into an SQL query, we need to break down the steps involved in calculating the median of the average monthly projected sales in USD for France in 2021. Here's a step-by-step breakdown:

1. **Filter Data**: Select records where `promo_total_id` is 1 and `channel_total_id` is 1.
2. **Calculate Growth Rate**: Apply the growth rate from 2019 to 2020 to project 2021 sales.
3. **Convert to USD**: Convert the projected sales to USD using the currency table.
4. **Calculate Monthly Averages**: Find the average monthly sales for each month in 2021.
5. **Determine Median**: Calculate the median of these monthly averages.

Here's the corresponding SQL query:

```sql
WITH ProjectedSales AS (
    SELECT 
        DATE_TRUNC('month', date) AS month,
        SUM(sales * growth_rate) AS projected_sales
    FROM 
        sales_data
    WHERE 
        promo_total_id = 1 AND channel_total_id = 1
    GROUP BY 
        DATE_TRUNC('month', date)
),
ConvertedSales AS (
    SELECT 
        month,
        projected_sales *