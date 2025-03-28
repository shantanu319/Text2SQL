```sql
SELECT 
    channel,
    COUNT(*) AS session_count
FROM 
    website_traffic
WHERE 
    date BETWEEN '2020-12-01' AND '2020-12-31'
GROUP BY 
    channel
ORDER BY 
    session_count DESC
LIMIT 4;
```

This query selects the channel with the fourth highest number of sessions from the `website_traffic` table for the month of December 2020. It groups the data by channel, counts the number of sessions for each channel, orders the results in descending order based on the session count, and limits the output to the top four channels.