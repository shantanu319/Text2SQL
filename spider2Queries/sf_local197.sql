To convert the given text into an SQL query, we need to assume a table structure that includes at least columns for customer ID, payment amount, and date. Let's assume the table is named `payments` with the following relevant columns:

- `customer_id`: The unique identifier for each customer.
- `payment_amount`: The amount paid by the customer.
- `payment_date`: The date when the payment was made.

Here's how you can write the SQL query to find the largest change in payment amounts from one month to the immediately following month:

```sql
WITH MonthlyPayments AS (
    SELECT 
        customer_id,
        payment_amount,
        DATE_TRUNC('month', payment_date) AS payment_month
    FROM 
        payments
),
MonthlyDifferences AS (
    SELECT 
        customer_id,
        payment_month,
        LAG(payment_amount) OVER (PARTITION BY customer_id ORDER BY payment_month) AS previous_month_payment,
        payment_amount - LAG(payment_amount) OVER (PARTITION BY customer_id ORDER BY payment_month) AS month_over_month_difference
    FROM 
        MonthlyPayments
)
SELECT 
    customer_id,
    payment_month,
    ROUND(month_over_month_difference, 2) AS max_monthly_difference
FROM 
    MonthlyDifferences