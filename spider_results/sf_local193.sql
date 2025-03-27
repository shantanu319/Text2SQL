```sql
SELECT 
    AVG(CASE WHEN DATEDIFF(T1.first_purchase_date, T2.purchase_date) <= 7 THEN T2.ltv ELSE NULL END) AS avg_ltv_first_7_days,
    AVG(CASE WHEN DATEDIFF(T1.first_purchase_date, T2.purchase_date) <= 30 THEN T2.ltv ELSE NULL END) AS avg_ltv_first_30_days,
    AVG(T2.ltv) AS avg_total_lifetime_sales
FROM 
    customers AS T1
JOIN 
    purchases AS T2 ON T1.customer_id = T2.customer_id
WHERE 
    T2.ltv > 0;
```

This SQL query calculates the average percentage of the total lifetime sales (LTV) that occur in the first 7 and 30 days after a customer's initial purchase, as well as the average total lifetime sales (LTV), for all customers with non-zero lifetime sales. It uses the `DATEDIFF` function to calculate the difference in days between the first purchase date and each subsequent purchase date, and then applies conditional logic to filter the results based on the specified time periods. The `AVG` function is used to compute the average values for each metric. <im_end