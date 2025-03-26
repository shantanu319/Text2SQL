SELECT MONTH(transaction_timestamp) AS month,
       YEAR(transaction_timestamp) AS year,
       COUNT(DISTINCT transaction_hash) AS total_transactions,
       ROUND(COUNT(*) / (EXTRACT(EPOCH FROM INTERVAL '1 month' * (MONTH(transaction_timestamp) - 1)) + EXTRACT(EPOCH FROM INTERVAL '1 day' * DAYOFMONTH(transaction_timestamp))), 2) AS transactions_per_second