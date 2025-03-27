To convert the given text into SQL, we need to break down the task into several steps and write corresponding SQL queries. Here's a step-by-step breakdown:

1. **Calculate Month-End Balance**: For each customer and each month of 2020, calculate the month-end balance by summing up all deposit amounts and subtracting all withdrawal amounts.

2. **Determine Highest and Lowest Count of Customers with Positive Month-End Balance**: Identify the months with the highest and lowest counts of customers who have a positive month-end balance.

3. **Compute Average Month-End Balance for Each of These Two Months**: Calculate the average month-end balance for the identified months.

4. **Find the Difference Between These Two Averages**: Compute the difference between the average month-end balances of the two months.

Here's the SQL query to achieve this:

```sql
WITH MonthlyBalances AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', transaction_date) AS transaction_month,
        SUM(CASE WHEN transaction_type = 'deposit' THEN amount ELSE -amount END) AS month_end_balance
    FROM 
        transactions
    WHERE 
        YEAR(transaction_date) = 2020
    GROUP BY 
        customer_id, transaction_month