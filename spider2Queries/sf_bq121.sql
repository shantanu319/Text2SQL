```sql
SELECT 
    YEAR(CURDATE()) - YEAR(joined) AS years_since_joined,
    AVG(reputation) AS avg_reputation,
    AVG(badges) AS avg_badges
FROM 
    stackoverflow_users
WHERE 
    joined <= '2021-10-01'
GROUP BY 
    years_since_joined;
```