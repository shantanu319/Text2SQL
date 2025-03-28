```sql
SELECT SUM(
    AVG(predicted_sales) OVER (
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ORDER BY date_column
    )
) AS total_5_day_symmetric_moving_average
FROM (
    SELECT 
        date_column,
        predicted_sales
    FROM 
        sales_data
    WHERE 
        date_column >= '2017-01-01' AND date_column <= '2018-08-29'
) AS subquery;
```

This SQL query calculates the 5-day symmetric moving average of predicted toy sales for December 5 to 8, 2018, using daily sales data from January 1, 2017, to August 29, 2018. The `predicted_sales` column is assumed to contain the predicted sales values, and `date_column` should be replaced with the actual column name that contains the dates. The result is then summed up to get the total of those four 5-day moving averages. Note that this assumes a simple linear regression model has been used to predict sales, which is not explicitly mentioned in the original question but is implied by the use of "predicted sales." Adjust the table