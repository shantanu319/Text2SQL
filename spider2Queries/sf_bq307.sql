```sql
SELECT 
    b.BadgeName,
    COUNT(u.UserID) AS NumberOfUsers,
    AVG(DATEDIFF(b.DateEarned, u.AccountCreationDate)) AS AverageDaysFromAccountCreation
FROM 
    Users u
JOIN 
    Badges b ON u.UserID = b.UserID AND b.BadgeType = 'Gold'
WHERE 
    b.DateEarned IS NOT NULL
GROUP BY 
    b.BadgeName
ORDER BY 
    NumberOfUsers DESC
LIMIT 10;
```