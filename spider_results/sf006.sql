To convert the given text into SQL, you need to create a query that calculates the number of active financial branch entities for each U.S. state on March 1, 2020, and December 31, 2021, and then computes the percentage change between these two dates. Here's how you can structure the SQL query:

```sql
SELECT 
    state,
    COUNT(CASE WHEN start_date <= '2020-03-01' AND (end_date IS NULL OR end_date >= '2020-03-01') THEN 1 END) AS active_2020,
    COUNT(CASE WHEN start_date <= '2021-12-31' AND (end_date IS NULL OR end_date >= '2021-12-31') THEN 1 END) AS active_2021,
    CASE 
        WHEN active_2020 = 0 THEN NULL
        ELSE ROUND((active_2021 - active_2020) * 100 / active_2020, 2)
    END AS percentage_change
FROM 
    financial_branch_entities
JOIN