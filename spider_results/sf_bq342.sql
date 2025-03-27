```sql
SELECT 
    AVG(hourly_change) AS avg_hourly_change_2019,
    AVG(hourly_change) AS avg_hourly_change_2020,
    AVG(hourly_change) - AVG(hourly_change) AS difference
FROM 
    ethereum_transactions
WHERE 
    token_address = '0x68e54af74b22acaccffa04ccaad13be16ed14eac'
    AND (sender_address = '0x8babf0ba311aab914c00e8fda7e8558a8b66de5d' OR receiver_address = '0xfbd6c6b112214d949dcdfb1217153bc0a742862f')
    AND transaction_date BETWEEN '2019-01-01' AND '2020-12-31';
```

This SQL query calculates the average hourly changes in transaction values for the specified Ethereum token within the given date range. It then compares these averages for 2019 and 2020 to