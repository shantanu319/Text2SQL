To achieve the desired result using SQL, you can break down the problem into several steps. Here's a step-by-step approach to writing the SQL queries:

1. **Calculate Daily Balances**: First, calculate the daily balances for each customer, including days without transactions by carrying forward the previous day's balance.

2. **Determine Highest Daily Balance per Month**: For each month, determine the highest daily balance each customer had during that month.

3. **Sum Maximum Daily Balances Across All Customers**: Finally, sum these maximum daily balances across all customers to obtain a monthly total.

Here's how you can write the SQL queries:

### Step 1: Calculate Daily Balances

Assuming you have a table named `transactions` with columns `customer_id`, `transaction_date`, and `amount`, you can use the following query to calculate the daily balances:

```sql
WITH CustomerTransactions AS (
    SELECT 
        customer_id,
        transaction_date,
        amount,
        LAG(amount) OVER (PARTITION BY customer_id ORDER BY transaction_date) AS prev_amount
    FROM 
        transactions
),
DailyBalances AS (
    SELECT 
        customer_id,
        transaction_date,
        CASE 
            WHEN prev_amount IS NULL THEN amount
            ELSE prev_amount +