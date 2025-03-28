```sql
SELECT 
    BankName,
    1 - CAST('% Insured (Estimated)' AS DECIMAL(5, 2)) AS PercentageOfUninsuredAssets
FROM 
    QuarterlyEstimates
WHERE 
    AssetsExceeding10Billion = 'Yes'
ORDER BY 
    PercentageOfUninsuredAssets DESC
LIMIT 10;
```

This SQL query selects the bank name and calculates the percentage of uninsured assets based on the estimated percentage from the `QuarterlyEstimates` table. It filters for banks that have assets exceeding $10 billion and orders them by the percentage of uninsured assets in descending order, limiting the result to the top 10 banks.