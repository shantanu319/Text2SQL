```sql
WITH UniqueTransactions AS (
    SELECT DISTINCT *
    FROM rev_transactions
),
ChannelGroupings AS (
    SELECT 
        channel_grouping,
        COUNT(DISTINCT country) AS num_countries
    FROM UniqueTransactions
    GROUP BY channel_grouping
    HAVING COUNT(DISTINCT country) > 1
),
CountryWithHighestTransactions AS (
    SELECT 
        channel_grouping,
        country,
        SUM(transactions) AS total_transactions
    FROM UniqueTransactions
    WHERE channel_grouping IN (SELECT channel_grouping FROM ChannelGroupings)
    GROUP BY channel_grouping, country
    ORDER BY total_transactions DESC
    LIMIT 1
)
SELECT 
    channel_grouping,
    country,
    total_transactions
FROM CountryWithHighestTransactions;
```