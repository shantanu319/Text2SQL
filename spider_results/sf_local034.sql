Certainly! To achieve this, you can use a combination of SQL functions and window functions. Here's an example query that calculates the average of the total number of payments made using the most preferred payment method for each product category:

```sql
WITH PreferredPaymentMethods AS (
    SELECT 
        ProductCategory,
        PaymentMethod,
        COUNT(*) AS TotalPayments
    FROM 
        Payments
    GROUP BY 
        ProductCategory, PaymentMethod
),
MostPreferredPaymentMethods AS (
    SELECT 
        ProductCategory,
        PaymentMethod,
        TotalPayments,
        ROW_NUMBER() OVER (PARTITION BY ProductCategory ORDER BY TotalPayments DESC) AS Rank
    FROM 
        PreferredPaymentMethods
)
SELECT 
    ProductCategory,
    AVG(TotalPayments) AS AverageTotalPayments
FROM 
    MostPreferredPaymentMethods
WHERE 
    Rank = 1
GROUP BY 
    ProductCategory;
```

### Explanation:
1. **PreferredPaymentMethods CTE**: This Common Table Expression (CTE) groups the payments by `ProductCategory` and `PaymentMethod`, counting the total number of payments for each combination.

2. **MostPreferredPaymentMethods CTE**: This CTE ranks the payment methods within each product category based on the total number of payments. The `