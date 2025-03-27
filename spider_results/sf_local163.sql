```sql
SELECT 
    rank,
    first_name,
    last_name,
    salary
FROM 
    faculty
WHERE 
    salary = (
        SELECT AVG(salary) 
        FROM faculty 
        WHERE rank = f.rank
    );
```